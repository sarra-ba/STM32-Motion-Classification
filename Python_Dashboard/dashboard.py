import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox, QGroupBox, QTextEdit)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
from collections import deque
import numpy as np
from datetime import datetime

class SensorTileVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_running = False
       
        # Buffers pour les données (2 secondes à 50Hz = 100 points)
        self.buffer_size = 100
        self.acc_x = deque(maxlen=self.buffer_size)
        self.acc_y = deque(maxlen=self.buffer_size)
        self.acc_z = deque(maxlen=self.buffer_size)
        self.gyro_x = deque(maxlen=self.buffer_size)
        self.gyro_y = deque(maxlen=self.buffer_size)
        self.gyro_z = deque(maxlen=self.buffer_size)
        self.time_data = deque(maxlen=self.buffer_size)
        self.time_counter = 0
       
        # Classes de mouvement
        self.movement_classes = {
            '0': 'Repos',
            '1': 'Circulaire',
            '2': 'Rectangulaire',
            '3': 'Rectiligne'
        }
       
        self.current_movement = 'En attente...'
       
        self.initUI()
       
    def initUI(self):
        self.setWindowTitle('SensorTile - Visualisation Temps Réel')
        self.setGeometry(100, 100, 1200, 800)
       
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
       
        # === Section de connexion ===
        connection_group = QGroupBox("Connexion Série")
        connection_layout = QHBoxLayout()
       
        self.port_combo = QComboBox()
        self.refresh_ports()
        connection_layout.addWidget(QLabel("Port COM:"))
        connection_layout.addWidget(self.port_combo)
       
        self.refresh_btn = QPushButton("Actualiser")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        connection_layout.addWidget(self.refresh_btn)
       
        self.connect_btn = QPushButton("Connecter")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn)
       
        connection_layout.addStretch()
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
       
        # === Section d'affichage du mouvement détecté ===
        movement_group = QGroupBox("Mouvement Détecté")
        movement_layout = QVBoxLayout()
       
        self.movement_label = QLabel(self.current_movement)
        self.movement_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.movement_label.setFont(font)
        self.movement_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        movement_layout.addWidget(self.movement_label)
       
        movement_group.setLayout(movement_layout)
        main_layout.addWidget(movement_group)
       
        # === Section des graphiques ===
        graphs_layout = QHBoxLayout()
       
        # Graphique Accéléromètre
        acc_widget = pg.PlotWidget(title="Accéléromètre (m/s²)")
        acc_widget.setLabel('left', 'Accélération', units='m/s²')
        acc_widget.setLabel('bottom', 'Temps', units='s')
        acc_widget.addLegend()
        acc_widget.setBackground('w')
        self.acc_x_curve = acc_widget.plot(pen=pg.mkPen('r', width=2), name='X')
        self.acc_y_curve = acc_widget.plot(pen=pg.mkPen('g', width=2), name='Y')
        self.acc_z_curve = acc_widget.plot(pen=pg.mkPen('b', width=2), name='Z')
        graphs_layout.addWidget(acc_widget)
       
        # Graphique Gyroscope
        gyro_widget = pg.PlotWidget(title="Gyroscope (°/s)")
        gyro_widget.setLabel('left', 'Vitesse angulaire', units='°/s')
        gyro_widget.setLabel('bottom', 'Temps', units='s')
        gyro_widget.addLegend()
        gyro_widget.setBackground('w')
        self.gyro_x_curve = gyro_widget.plot(pen=pg.mkPen('r', width=2), name='X')
        self.gyro_y_curve = gyro_widget.plot(pen=pg.mkPen('g', width=2), name='Y')
        self.gyro_z_curve = gyro_widget.plot(pen=pg.mkPen('b', width=2), name='Z')
        graphs_layout.addWidget(gyro_widget)
       
        main_layout.addLayout(graphs_layout)
       
        # === Section des statistiques ===
        stats_group = QGroupBox("Statistiques en Temps Réel")
        stats_layout = QHBoxLayout()
       
        self.stats_label = QLabel("En attente de données...")
        self.stats_label.setStyleSheet("padding: 10px;")
        stats_layout.addWidget(self.stats_label)
       
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
       
        # === Section des données brutes reçues ===
        data_group = QGroupBox("Données Série Reçues (Brutes)")
        data_layout = QVBoxLayout()
       
        # Zone de texte pour afficher les données brutes
        from PyQt5.QtWidgets import QTextEdit
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setMaximumHeight(150)
        self.data_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                padding: 5px;
            }
        """)
        data_layout.addWidget(self.data_display)
       
        # Compteur de lignes
        self.line_counter_label = QLabel("Lignes reçues: 0")
        self.line_counter_label.setStyleSheet("padding: 5px; font-weight: bold;")
        data_layout.addWidget(self.line_counter_label)
       
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)
       
        # Compteur de lignes
        self.line_counter = 0
       
        # Timer pour la lecture série
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial_data)
       
    def refresh_ports(self):
        """Actualise la liste des ports COM disponibles"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
   
    def toggle_connection(self):
        """Gère la connexion/déconnexion du port série"""
        if not self.is_running:
            self.connect_serial()
        else:
            self.disconnect_serial()
   
    def connect_serial(self):
        """Établit la connexion série"""
        try:
            port_text = self.port_combo.currentText()
            if not port_text:
                # Si aucun port sélectionné, essayer COM6 par défaut
                port = "COM6"
            else:
                port = port_text.split(' - ')[0]
           
            print(f"🔌 Tentative de connexion à {port}...")
           
            # Configuration de la connexion série (ajuster selon votre configuration)
            self.serial_port = serial.Serial(
                port=port,
                baudrate=115200,  # Ajuster selon votre configuration
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
           
            # Vider le buffer
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
           
            self.is_running = True
            self.connect_btn.setText("Déconnecter")
            self.connect_btn.setStyleSheet("background-color: #e74c3c;")
            self.timer.start(50)  # Lecture toutes les 50ms (20Hz)
            self.update_status(f"Connecté à {port}", "#27ae60")
           
            print(f"✓ Connexion établie sur {port}")
            print(f"✓ Baudrate: 115200")
            print(f"✓ En attente de données...")
            print(f"✓ Bytes en attente: {self.serial_port.in_waiting}")
           
        except Exception as e:
            self.update_status(f"Erreur: {str(e)}", "red")
            print(f"❌ Erreur de connexion: {str(e)}")
            print(f"💡 Vérifiez que:")
            print(f"   - Le SensorTile est bien connecté")
            print(f"   - Le bon port COM est sélectionné")
            print(f"   - Aucun autre programme n'utilise le port")
            print(f"   - Le baudrate correspond (115200)")
   
    def disconnect_serial(self):
        """Ferme la connexion série"""
        self.is_running = False
        self.timer.stop()
       
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
       
        self.connect_btn.setText("Connecter")
        self.connect_btn.setStyleSheet("")
        self.update_status("Déconnecté", "gray")
   
    def read_serial_data(self):
        """Lit et traite les données du port série"""
        if not self.serial_port or not self.serial_port.is_open:
            return
       
        try:
            # Lire toutes les lignes disponibles
            while self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
               
                if line:
                    # Afficher dans la console de debug
                    print(f"📥 Reçu: '{line}'")
                   
                    # Afficher dans l'interface graphique
                    self.display_raw_data(line)
                   
                    # Parser les données
                    self.parse_data(line)
                else:
                    print("⚠ Ligne vide reçue")
                   
        except UnicodeDecodeError as e:
            print(f"⚠ Erreur d'encodage: {e}")
            # Essayer avec latin-1
            try:
                line = self.serial_port.readline().decode('latin-1', errors='ignore').strip()
                if line:
                    print(f"📥 Reçu (latin-1): '{line}'")
                    self.display_raw_data(line)
                    self.parse_data(line)
            except:
                pass
        except Exception as e:
            print(f"❌ Erreur de lecture série: {e}")
   
    def display_raw_data(self, line):
        """Affiche les données brutes dans l'interface"""
        from datetime import datetime
       
        # Incrémenter le compteur
        self.line_counter += 1
        self.line_counter_label.setText(f"Lignes reçues: {self.line_counter}")
       
        # Ajouter timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_line = f"[{timestamp}] {line}"
       
        # Ajouter au QTextEdit
        self.data_display.append(formatted_line)
       
        # Auto-scroll vers le bas
        scrollbar = self.data_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
       
        # Limiter le nombre de lignes affichées (garder les 100 dernières)
        if self.line_counter % 100 == 0:
            # Garder seulement les dernières lignes
            text = self.data_display.toPlainText()
            lines = text.split('\n')
            if len(lines) > 100:
                self.data_display.setPlainText('\n'.join(lines[-100:]))
   
    def parse_data(self, line):
        """
        Parse les données reçues du SensorTile
        Détecte automatiquement différents formats possibles
        """
        try:
            print(f"🔍 Analyse: '{line}' (longueur: {len(line)})")  # Debug détaillé
           
            # Suppression des espaces
            line = line.strip()
           
            if not line:
                print("⚠ Ligne vide après strip")
                return
           
            # Format 1: "Orientation: left" ou "Movement: Circulaire"
            if ':' in line and ',' not in line:
                print("📋 Format détecté: Texte avec ':' ")
                parts = line.split(':')
                if len(parts) == 2:
                    label = parts[1].strip().lower()
                    print(f"   Label trouvé: '{label}'")
                    # Mapper les orientations/mouvements aux classes
                    movement_map = {
                        'repos': '0', 'rest': '0', 'immobile': '0',
                        'circulaire': '1', 'circular': '1', 'circle': '1',
                        'rectangulaire': '2', 'rectangular': '2', 'rectangle': '2',
                        'rectiligne': '3', 'linear': '3', 'straight': '3',
                        'left': '1', 'right': '1', 'up': '2', 'down': '2',
                        'forward': '3', 'backward': '3'
                    }
                    movement_class = movement_map.get(label, '0')
                    print(f"   ✓ Classe mappée: {movement_class}")
                    self.update_movement(movement_class)
                    return
           
            # Format 2: "CLASS,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z"
            if ',' in line:
                print("📋 Format détecté: CSV")
                parts = line.split(',')
                print(f"   Nombre de champs: {len(parts)}")
               
                if len(parts) >= 7:
                    print("   ✓ Format complet (7+ champs)")
                    movement_class = parts[0].strip()
                    acc_x_val = float(parts[1])
                    acc_y_val = float(parts[2])
                    acc_z_val = float(parts[3])
                    gyro_x_val = float(parts[4])
                    gyro_y_val = float(parts[5])
                    gyro_z_val = float(parts[6])
                   
                    print(f"   ACC: [{acc_x_val:.2f}, {acc_y_val:.2f}, {acc_z_val:.2f}]")
                    print(f"   GYRO: [{gyro_x_val:.2f}, {gyro_y_val:.2f}, {gyro_z_val:.2f}]")
                   
                    # Mise à jour des buffers
                    self.time_counter += 0.02  # 50Hz
                    self.time_data.append(self.time_counter)
                    self.acc_x.append(acc_x_val)
                    self.acc_y.append(acc_y_val)
                    self.acc_z.append(acc_z_val)
                    self.gyro_x.append(gyro_x_val)
                    self.gyro_y.append(gyro_y_val)
                    self.gyro_z.append(gyro_z_val)
                   
                    # Mise à jour du mouvement détecté
                    self.update_movement(movement_class)
                   
                    # Mise à jour des graphiques
                    self.update_plots()
                   
                    # Mise à jour des statistiques
                    self.update_statistics()
                    print("   ✓ Graphiques mis à jour")
               
                # Format 3: Seulement données capteurs "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z"
                elif len(parts) == 6:
                    print("   ✓ Format capteurs uniquement (6 champs)")
                    acc_x_val = float(parts[0])
                    acc_y_val = float(parts[1])
                    acc_z_val = float(parts[2])
                    gyro_x_val = float(parts[3])
                    gyro_y_val = float(parts[4])
                    gyro_z_val = float(parts[5])
                   
                    # Mise à jour des buffers
                    self.time_counter += 0.02
                    self.time_data.append(self.time_counter)
                    self.acc_x.append(acc_x_val)
                    self.acc_y.append(acc_y_val)
                    self.acc_z.append(acc_z_val)
                    self.gyro_x.append(gyro_x_val)
                    self.gyro_y.append(gyro_y_val)
                    self.gyro_z.append(gyro_z_val)
                   
                    # Mise à jour des graphiques
                    self.update_plots()
                    self.update_statistics()
                    print("   ✓ Graphiques mis à jour")
                else:
                    print(f"   ⚠ Nombre de champs non supporté: {len(parts)}")
           
            # Format 4: Juste un numéro de classe "0", "1", "2", "3"
            elif line.isdigit():
                print(f"📋 Format détecté: Numéro simple '{line}'")
                self.update_movement(line)
                print("   ✓ Mouvement mis à jour")
            else:
                print(f"⚠ Format non reconnu: '{line}'")
               
        except ValueError as e:
            print(f"❌ Erreur de conversion: {e}")
            print(f"   Ligne problématique: '{line}'")
        except Exception as e:
            print(f"❌ Erreur de parsing: {e}")
            print(f"   Ligne: '{line}'")
   
    def update_movement(self, movement_class):
        """Met à jour l'affichage du mouvement détecté"""
        movement_name = self.movement_classes.get(movement_class, 'Inconnu')
       
        if movement_name != self.current_movement:
            self.current_movement = movement_name
            self.movement_label.setText(movement_name)
           
            # Changement de couleur selon le mouvement
            colors = {
                'Repos': '#95a5a6',
                'Circulaire': '#3498db',
                'Rectangulaire': '#e74c3c',
                'Rectiligne': '#2ecc71',
                'En attente...': '#2c3e50',
                'Inconnu': '#f39c12'
            }
           
            color = colors.get(movement_name, '#2c3e50')
            self.movement_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                }}
            """)
   
    def update_plots(self):
        """Met à jour les graphiques en temps réel"""
        if len(self.time_data) > 1:
            time_array = np.array(self.time_data)
           
            # Mise à jour Accéléromètre
            self.acc_x_curve.setData(time_array, np.array(self.acc_x))
            self.acc_y_curve.setData(time_array, np.array(self.acc_y))
            self.acc_z_curve.setData(time_array, np.array(self.acc_z))
           
            # Mise à jour Gyroscope
            self.gyro_x_curve.setData(time_array, np.array(self.gyro_x))
            self.gyro_y_curve.setData(time_array, np.array(self.gyro_y))
            self.gyro_z_curve.setData(time_array, np.array(self.gyro_z))
   
    def update_statistics(self):
        """Met à jour les statistiques affichées"""
        if len(self.acc_x) > 0:
            acc_magnitude = np.sqrt(
                np.array(self.acc_x)**2 +
                np.array(self.acc_y)**2 +
                np.array(self.acc_z)**2
            )
            gyro_magnitude = np.sqrt(
                np.array(self.gyro_x)**2 +
                np.array(self.gyro_y)**2 +
                np.array(self.gyro_z)**2
            )
           
            stats_text = f"""
            <b>Accéléromètre:</b> |A| = {acc_magnitude[-1]:.2f} m/s²
            (Moy: {np.mean(acc_magnitude):.2f}, Max: {np.max(acc_magnitude):.2f})
            &nbsp;&nbsp;&nbsp;
            <b>Gyroscope:</b> |G| = {gyro_magnitude[-1]:.2f} °/s
            (Moy: {np.mean(gyro_magnitude):.2f}, Max: {np.max(gyro_magnitude):.2f})
            """
           
            self.stats_label.setText(stats_text)
   
    def update_status(self, message, color):
        """Affiche un message de statut"""
        self.setWindowTitle(f'SensorTile - {message}')
   
    def closeEvent(self, event):
        """Gère la fermeture de l'application"""
        self.disconnect_serial()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
   
    # Style de l'application
    app.setStyle('Fusion')
   
    window = SensorTileVisualizer()
    window.show()
   
    sys.exit(app.exec_())
