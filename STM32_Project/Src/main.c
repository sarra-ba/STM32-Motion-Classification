/**
******************************************************************************
* @file    main.c
* @brief   AI Movement Classification
******************************************************************************
*/

#include <string.h>
#include <stdio.h>
#include <math.h>
#include "main.h"
#include "datalog_application.h"
#include "usbd_cdc_interface.h"
#include "ff_gen_drv.h"
#include "sd_diskio.h"

/* AI includes */
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

#define DATA_PERIOD_MS (100)
#define WINDOW_SIZE 100
#define NUM_FEATURES 6

uint8_t SendOverUSB = 1;
USBD_HandleTypeDef USBD_Device;
static volatile uint8_t MEMSInterrupt = 0;

/* AI buffers */
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_float in_data[AI_NETWORK_IN_1_SIZE];
static ai_float out_data[AI_NETWORK_OUT_1_SIZE];
static ai_handle network = AI_HANDLE_NULL;
static ai_buffer *ai_input = NULL;
static ai_buffer *ai_output = NULL;

static volatile uint8_t isCollecting = 0;
static int sampleCount = 0;
static float sensor_buffer[WINDOW_SIZE][NUM_FEATURES];
static const char* class_names[] = {"Circular", "Rectangular", "Linear"};
static uint8_t ai_ready = 0;

static RTC_HandleTypeDef RtcHandle;
static void *LSM6DSM_X_0_handle = NULL;
static void *LSM6DSM_G_0_handle = NULL;
static void *LSM303AGR_X_0_handle = NULL;
static void *LSM303AGR_M_0_handle = NULL;
static void *LPS22HB_P_0_handle = NULL;
static void *LPS22HB_T_0_handle = NULL;
static void *HTS221_H_0_handle = NULL;
static void *HTS221_T_0_handle = NULL;
static void *GG_handle = NULL;

static void Error_Handler(void);
static void RTC_Config(void);
static void RTC_TimeStampConfig(void);
static void initializeAllSensors(void);
void enableAllSensors(void);
void disableAllSensors(void);

static uint8_t AI_Init(void)
{
    char msg[64];

    // Create network
    ai_error err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        sprintf(msg, "AI create error: %d\r\n", err.type);
        CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
        return 0;
    }

    // Prepare network parameters (flat initialization)
    ai_network_params params = {
        AI_NETWORK_DATA_WEIGHTS(s_network_weights_array_u64),
        AI_NETWORK_DATA_ACTIVATIONS(activations)
    };

    // Initialize network
    ai_bool r = ai_network_init(network, &params);
    if (!r) {
        CDC_Fill_Buffer((uint8_t*)"AI init failed\r\n", 15);
        return 0;
    }

    // Get input/output buffers
    ai_input = ai_network_inputs_get(network, NULL);
    ai_output = ai_network_outputs_get(network, NULL);
    if (!ai_input || !ai_output) {
        CDC_Fill_Buffer((uint8_t*)"AI input/output NULL\r\n", 22);
        return 0;
    }

    CDC_Fill_Buffer((uint8_t*)"AI init success\r\n", 17);
    return 1;
}


/* Run AI */
static void RunAI(void)
{
    int i, ch;
    char msg[128];

    // Normalize sensor data
    for(ch = 0; ch < NUM_FEATURES; ch++)
    {
        float sum = 0.0f;
        for(i = 0; i < WINDOW_SIZE; i++)
            sum += sensor_buffer[i][ch];
        float mean = sum / WINDOW_SIZE;

        float sum_sq = 0.0f;
        for(i = 0; i < WINDOW_SIZE; i++)
        {
            float d = sensor_buffer[i][ch] - mean;
            sum_sq += d * d;
        }
        float std = sqrtf(sum_sq / WINDOW_SIZE);
        if(std < 0.000001f) std = 1.0f;

        for(i = 0; i < WINDOW_SIZE; i++)
            in_data[i * NUM_FEATURES + ch] = (sensor_buffer[i][ch] - mean) / std;
    }

    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    ai_i32 n = ai_network_run(network, ai_input, ai_output);
    HAL_Delay(500);

    if(n != 1) {
        CDC_Fill_Buffer((uint8_t*)"AI run ERR\r\n\r\n", 14);
        return;
    }

    // DEBUG: Print raw output values
    sprintf(msg, "Raw outputs:\r\n");
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(100);

    sprintf(msg, "  Circular:    %.4f\r\n", out_data[0]);
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(100);

    sprintf(msg, "  Rectangular: %.4f\r\n", out_data[1]);
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(100);

    sprintf(msg, "  Linear:      %.4f\r\n\r\n", out_data[2]);
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(100);

    // Apply SOFTMAX to convert logits to probabilities
    float max_val = out_data[0];
    for(i = 1; i < 3; i++)
        if(out_data[i] > max_val)
            max_val = out_data[i];

    // Subtract max for numerical stability
    float exp_sum = 0.0f;
    float exp_vals[3];
    for(i = 0; i < 3; i++)
    {
        exp_vals[i] = expf(out_data[i] - max_val);
        exp_sum += exp_vals[i];
    }

    // Convert to probabilities
    float probs[3];
    for(i = 0; i < 3; i++)
        probs[i] = exp_vals[i] / exp_sum;

    // Find max probability
    int max_i = 0;
    for(i = 1; i < 3; i++)
        if(probs[i] > probs[max_i])
            max_i = i;

    sprintf(msg, "=== %s (%.0f%%) ===\r\n\r\n",
            class_names[max_i], probs[max_i] * 100.0f);
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));

    // ✅ AJOUTÉ : Envoyer le mouvement détecté pour le dashboard
    sprintf(msg, "Movement: %s\r\n", class_names[max_i]);
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
}

/* Collect Data */
void CollectData(void *acc, void *gyr)
{
    SensorAxes_t a, g;
    char line[128];  // ✅ AJOUTÉ

    BSP_ACCELERO_Get_Axes(acc, &a);
    BSP_GYRO_Get_Axes(gyr, &g);

    sensor_buffer[sampleCount][0] = (float)a.AXIS_X;
    sensor_buffer[sampleCount][1] = (float)a.AXIS_Y;
    sensor_buffer[sampleCount][2] = (float)a.AXIS_Z;
    sensor_buffer[sampleCount][3] = (float)g.AXIS_X;
    sensor_buffer[sampleCount][4] = (float)g.AXIS_Y;
    sensor_buffer[sampleCount][5] = (float)g.AXIS_Z;

    // ✅ AJOUTÉ : Envoyer les données pour le dashboard (format CSV)
    sprintf(line, "%d,%d,%d,%d,%d,%d\r\n",
            a.AXIS_X, a.AXIS_Y, a.AXIS_Z,
            g.AXIS_X, g.AXIS_Y, g.AXIS_Z);
    CDC_Fill_Buffer((uint8_t*)line, strlen(line));
}

/* Main */
int main(void)
{
    uint32_t msTick, msTickPrev = 0;
    uint8_t doubleTap = 0;
    char msg[256];

    HAL_Init();
    SystemClock_Config();

    BSP_LED_Init(LED1);
    BSP_LED_On(LED1);

    RTC_Config();
    RTC_TimeStampConfig();
    HAL_PWREx_EnableVddUSB();

    USBD_Init(&USBD_Device, &VCP_Desc, 0);
    USBD_RegisterClass(&USBD_Device, USBD_CDC_CLASS);
    USBD_CDC_RegisterInterface(&USBD_Device, &USBD_CDC_fops);
    USBD_Start(&USBD_Device);

    HAL_Delay(3000);

    Sensor_IO_SPI_CS_Init_All();
    initializeAllSensors();
    enableAllSensors();

    sprintf(msg, "\r\n\r\n========================================\r\n");
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(200);

    sprintf(msg, "  AI CLASSIFICATION\r\n");
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(200);

    sprintf(msg, "========================================\r\n\r\n");
    CDC_Fill_Buffer((uint8_t*)msg, strlen(msg));
    HAL_Delay(200);

    ai_ready = AI_Init();

    if(ai_ready)
        CDC_Fill_Buffer((uint8_t*)"AI Ready!\r\n\r\n", 13);
    else
        CDC_Fill_Buffer((uint8_t*)"AI Failed!\r\n\r\n", 14);

    HAL_Delay(200);

    CDC_Fill_Buffer((uint8_t*)"Double-tap to classify\r\n\r\n", 27);
    HAL_Delay(200);

    while (1)
    {
        msTick = HAL_GetTick();

        if(msTick % DATA_PERIOD_MS == 0 && msTickPrev != msTick)
        {
            msTickPrev = msTick;
            BSP_LED_Toggle(LED1);

            if(isCollecting && sampleCount < WINDOW_SIZE)
            {
                CollectData(LSM6DSM_X_0_handle, LSM6DSM_G_0_handle);
                sampleCount++;

                if(sampleCount % 20 == 0)
                {
                    char p[64];
                    sprintf(p, "[%d/%d]\r\n", sampleCount, WINDOW_SIZE);
                    CDC_Fill_Buffer((uint8_t*)p, strlen(p));
                }

                if(sampleCount >= WINDOW_SIZE)
                {
                    CDC_Fill_Buffer((uint8_t*)"Processing...\r\n", 15);
                    HAL_Delay(200);

                    if(ai_ready)
                        RunAI();

                    isCollecting = 0;
                    sampleCount = 0;
                    BSP_LED_On(LED1);
                }
            }
        }

        BSP_ACCELERO_Get_Double_Tap_Detection_Status_Ext(LSM6DSM_X_0_handle, &doubleTap);

        if(doubleTap && !isCollecting && ai_ready)
        {
            isCollecting = 1;
            sampleCount = 0;
            doubleTap = 0;  // ✅ AJOUTÉ : Reset le flag pour permettre multiple détections

            CDC_Fill_Buffer((uint8_t*)"\r\nCollecting...\r\n", 17);
            HAL_Delay(100);

            BSP_LED_Off(LED1);
        }

        HAL_Delay(1);
    }
}

/* Sensor initialization functions */
static void initializeAllSensors(void)
{
    if (BSP_ACCELERO_Init(LSM6DSM_X_0, &LSM6DSM_X_0_handle) != COMPONENT_OK) while(1);
    if (BSP_GYRO_Init(LSM6DSM_G_0, &LSM6DSM_G_0_handle) != COMPONENT_OK) while(1);
    if (BSP_ACCELERO_Init(LSM303AGR_X_0, &LSM303AGR_X_0_handle) != COMPONENT_OK) while(1);
    if (BSP_MAGNETO_Init(LSM303AGR_M_0, &LSM303AGR_M_0_handle) != COMPONENT_OK) while(1);
    if (BSP_PRESSURE_Init(LPS22HB_P_0, &LPS22HB_P_0_handle) != COMPONENT_OK) while(1);
    if (BSP_TEMPERATURE_Init(LPS22HB_T_0, &LPS22HB_T_0_handle) != COMPONENT_OK) while(1);
    BSP_ACCELERO_Enable_Double_Tap_Detection_Ext(LSM6DSM_X_0_handle);
    BSP_ACCELERO_Set_Tap_Threshold_Ext(LSM6DSM_X_0_handle, LSM6DSM_TAP_THRESHOLD_MID);
}

void enableAllSensors(void)
{
    BSP_ACCELERO_Sensor_Enable(LSM6DSM_X_0_handle);
    BSP_GYRO_Sensor_Enable(LSM6DSM_G_0_handle);
    BSP_ACCELERO_Sensor_Enable(LSM303AGR_X_0_handle);
    BSP_MAGNETO_Sensor_Enable(LSM303AGR_M_0_handle);
    BSP_PRESSURE_Sensor_Enable(LPS22HB_P_0_handle);
    BSP_TEMPERATURE_Sensor_Enable(LPS22HB_T_0_handle);
}

void disableAllSensors(void)
{
    BSP_ACCELERO_Sensor_Disable(LSM6DSM_X_0_handle);
    BSP_ACCELERO_Sensor_Disable(LSM303AGR_X_0_handle);
    BSP_GYRO_Sensor_Disable(LSM6DSM_G_0_handle);
    BSP_MAGNETO_Sensor_Disable(LSM303AGR_M_0_handle);
    BSP_PRESSURE_Sensor_Disable(LPS22HB_P_0_handle);
    BSP_TEMPERATURE_Sensor_Disable(LPS22HB_T_0_handle);
}

/* RTC configuration */
static void RTC_Config(void)
{
    RtcHandle.Instance = RTC;
    RtcHandle.Init.HourFormat = RTC_HOURFORMAT_12;
    RtcHandle.Init.AsynchPrediv = RTC_ASYNCH_PREDIV;
    RtcHandle.Init.SynchPrediv = RTC_SYNCH_PREDIV;
    RtcHandle.Init.OutPut = RTC_OUTPUT_DISABLE;
    RtcHandle.Init.OutPutPolarity = RTC_OUTPUT_POLARITY_HIGH;
    RtcHandle.Init.OutPutType = RTC_OUTPUT_TYPE_OPENDRAIN;
    if (HAL_RTC_Init(&RtcHandle) != HAL_OK) Error_Handler();
}

static void RTC_TimeStampConfig(void)
{
    RTC_DateTypeDef sd;
    RTC_TimeTypeDef st;
    sd.Year = 0x00;
    sd.Month = RTC_MONTH_JANUARY;
    sd.Date = 0x01;
    sd.WeekDay = RTC_WEEKDAY_MONDAY;
    if (HAL_RTC_SetDate(&RtcHandle, &sd, FORMAT_BCD) != HAL_OK) Error_Handler();
    st.Hours = 0x00;
    st.Minutes = 0x00;
    st.Seconds = 0x00;
    st.TimeFormat = RTC_HOURFORMAT12_AM;
    st.DayLightSaving = RTC_DAYLIGHTSAVING_NONE;
    st.StoreOperation = RTC_STOREOPERATION_RESET;
    if (HAL_RTC_SetTime(&RtcHandle, &st, FORMAT_BCD) != HAL_OK) Error_Handler();
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    MEMSInterrupt = 1;
}

static void Error_Handler(void)
{
    while(1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    while(1) {}
}
#endif
