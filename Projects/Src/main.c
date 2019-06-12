/**
  ******************************************************************************
  * @file    Display/LCD_PicturesFromSDCard/Src/main.c
  * @author  MCD Application Team
  * @brief   This file provides main program functions
  ************
	******************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics International N.V.
  * All rights reserved.</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without 
  * modification, are permitted, provided that the following conditions are met:+
  *
  * 1. Redistribution of source code must retain the above copyright notice, 
  *    this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  *    this list of conditions and the following disclaimer in the documentation
  *    and/or other materials provided with the distribution.
  * 3. Neither the name of STMicroelectronics nor the names of other 
  *    contributors to this software may be used to endorse or promote products 
  *    derived from this software without specific written permission.
  * 4. This software, including modifications and/or derivative works of this 
  *    software, must execute solely and exclusively on microcontroller or
  *    microprocessor devices manufactured by or for STMicroelectronics.
  * 5. Redistribution and use of this software other than as permitted under 
  *    this license is void and will automatically terminate your rights under 
  *    this license. 
  *
  * THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS" 
  * AND ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT 
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
  * PARTICULAR PURPOSE AND NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY
  * RIGHTS ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT 
  * SHALL STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
  * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "mfcc.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define OUT_DIM 				2
#define SAMP_FREQ 			8000
#define FRAME_SHIFT_MS 	20
#define FRAME_LENGTH_MS 32

#define NUM_FBANK_BINS 	40
#define NUM_MFCC_COEFFS 10
#define MEL_LOW_FREQ 		20
#define MEL_HIGH_FREQ 	4000

#define FRAME_SHIFT ((int16_t)(SAMP_FREQ * 0.001 * FRAME_SHIFT_MS))
#define FRAME_LEN ((int16_t)(SAMP_FREQ * 0.001 * FRAME_LENGTH_MS))

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
FATFS SD_FatFs;  /* File system object for SD card logical drive */
char SD_Path[4]; /* SD card logical drive path */
char* pDirectoryFiles[MAX_BMP_FILES];
char *word_list[] = {"unkown","xiaozhi"};
char mfcc_features[990];
short pcm_data[320];
FATFS fatfs;
FIL wav_file, conf_file;
/* Internal Buffer defined in SDRAM memory */
uint8_t *uwInternalBuffer;

/* Private function prototypes -----------------------------------------------*/
static void LCD_Config(void);
static void SystemClock_Config(void);
static void Error_Handler(void);
static void CPU_CACHE_Enable(void);;
int tengine_test_init(void * output_buf,int output_size);
void *tengine_test_run(const void * input, int input_size);
void tengine_test_free(void);
void dump_mem_stat(void);	

/* Private functions ---------------------------------------------------------*/


/**
  * @brief  Main program
  * @param  None
  * @retval None
  */
int main(void)
{
  uint32_t counter = 0, transparency = 0;
  uint8_t str[30];
  uwInternalBuffer = (uint8_t *)INTERNAL_BUFFER_START_ADDRESS;
  
  /* Enable the CPU Cache */
  CPU_CACHE_Enable();
  
  /* STM32F7xx HAL library initialization:
       - Configure the Flash ART accelerator on ITCM interface
       - Configure the Systick to generate an interrupt each 1 msec
       - Set NVIC Group Priority to 4
       - Global MSP (MCU Support Package) initialization
     */
  HAL_Init();

  /* Configure the system clock to 180 MHz */
  SystemClock_Config();

  /* Configure LED2 */
  BSP_LED_Init(LED2);

  /*##-1- Configure LCD ######################################################*/
  LCD_Config();

  /*##-2- Link the SD Card disk I/O driver ###################################*/
  if(FATFS_LinkDriver(&SD_Driver, SD_Path) == 0)
  {
		/* Open filesystem */
    if(f_mount(&fatfs, (TCHAR const*)"",1) != FR_OK)
    {
      printf("failed to mount file system!\n");
    }
  }
  else
  {
    /* FatFs Initialization Error */
    Error_Handler();
  }

	
	char score[OUT_DIM];
	tengine_test_init(score,2);
	MFCC_init();
	FRESULT ret = f_open (&conf_file, "conf.txt", FA_CREATE_ALWAYS|FA_WRITE);
	if(ret != FR_OK)
	{
		printf("failed to open conf_list.txt, ret=%d\n", ret);
		goto exit;
	}
	
	
	for(int i=0; i<6; i++)
	{
		memset(mfcc_features, 0, 990);
		char file_name[20];
		sprintf(file_name, "%d.wav", i);
		FRESULT ret = f_open (&wav_file, file_name, FA_READ);
		if(ret != FR_OK)
		{
			printf("failed to open %s, ret=%d\n", file_name, ret);
			continue;
		}
		
		UINT readed_len = 0;
		//read and drop wav head 44 bytes
		ret = f_read (&wav_file, pcm_data, 44, &readed_len);
		if(ret != FR_OK)
		{
			printf("failed to read 0:/0.wav, ret=%d\n", ret);
		}
		
		memset(pcm_data, 0, 640);
		int pre_data_len = (FRAME_LEN-FRAME_SHIFT)*2;
		ret = f_read (&wav_file, pcm_data, pre_data_len, &readed_len);
		if(ret != FR_OK)
		{
			printf("failed to read 0:/0.wav, ret=%d\n", ret);

		}
		//printf("failed to read 0");
		int mfcc_index = 0;
	

		for(int j=0; j<99; j++)
		{
			if(f_read (&wav_file, pcm_data+320-FRAME_SHIFT, FRAME_SHIFT*2, &readed_len) != FR_OK)
			{
				memset(pcm_data+320-FRAME_SHIFT, 0, FRAME_SHIFT*2);
			}
			
			MFCC_mfcc_compute(pcm_data, 0, &mfcc_features[mfcc_index]);
			memmove(pcm_data, pcm_data+FRAME_SHIFT, FRAME_SHIFT*2);
			mfcc_index += 10;

	  }
		f_close(&wav_file);

		
		int wakeup_index = 0;
		if(score[0] < score[1])
		{
			wakeup_index = 1;
		}

		
		score[0]=1;
		score[1]=1;
		
		printf("\nTengine Test: %s \n",file_name);

		
		char * tscore=tengine_test_run(mfcc_features,990);
		
		printf("tengine score = %d %d\n", score[0],score[1]);

		
		char output_info[100];
		sprintf(output_info, "%s Conf: ", file_name);
		UINT wirte_len = 0;
		f_write(&conf_file, output_info, strlen(output_info), &wirte_len);
		for(int i=0; i<OUT_DIM; i++)
		{
			sprintf(output_info, "%s=[%d], ", word_list[i], score[i]);
			f_write(&conf_file, output_info, strlen(output_info), &wirte_len);
		}
		f_write(&conf_file, "\n", 1, &wirte_len);
		f_sync(&conf_file);	
		memset(score, 0, 2);

  }
exit:
	
	tengine_test_free();
	
  /* Main infinite loop */
  while(1)
  {
		/* Insert a 1s delay */
    HAL_Delay(1000);
    
    /* Toggle LED2 */
    BSP_LED_Toggle(LED2);
  }
}

/**
  * @brief  LCD configuration
  * @param  None
  * @retval None
  */
static void LCD_Config(void)
{
  uint8_t lcd_status = LCD_OK;
  
  /* LCD DSI initialization in mode Video Burst */
  /* Initialize DSI LCD */
  BSP_LCD_Init();
  while(lcd_status != LCD_OK);
  
  BSP_LCD_LayerDefaultInit(LTDC_ACTIVE_LAYER_FOREGROUND, LCD_FB_START_ADDRESS);
  
  
  /* Select the LCD Foreground Layer */
  BSP_LCD_SelectLayer(LTDC_ACTIVE_LAYER_FOREGROUND);
  
  /* Clear the Foreground Layer */
  BSP_LCD_Clear(LCD_COLOR_WHITE);
  BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
	BSP_LCD_SetTextColor(LCD_COLOR_DARKBLUE);
	BSP_LCD_SetFont (&Font20);
		
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
static void Error_Handler(void)
{
  /* Turn LED1 on */
  BSP_LED_On(LED1);
  while(1)
  {
  }
}

/**
  * @brief  CPU L1-Cache enable.
  * @param  None
  * @retval None
  */
static void CPU_CACHE_Enable(void)
{
  /* Enable I-Cache */
  SCB_EnableICache();

  /* Enable D-Cache */
  SCB_EnableDCache();
}

/**
  * @brief  System Clock Configuration
  *         The system Clock is configured as follow :
  *            System Clock source            = PLL (HSE)
  *            SYSCLK(Hz)                     = 180000000
  *            HCLK(Hz)                       = 180000000
  *            AHB Prescaler                  = 1
  *            APB1 Prescaler                 = 4
  *            APB2 Prescaler                 = 2
  *            HSE Frequency(Hz)              = 25000000
  *            PLL_M                          = 25
  *            PLL_N                          = 360
  *            PLL_P                          = 2
  *            PLL_Q                          = 7
  *            PLL_R                          = 6
  *            VDD(V)                         = 3.3
  *            Main regulator output voltage  = Scale1 mode
  *            Flash Latency(WS)              = 5
  * @param  None
  * @retval None
  */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  HAL_StatusTypeDef ret = HAL_OK;

  /* Enable Power Control clock */
  __HAL_RCC_PWR_CLK_ENABLE();

  /* The voltage scaling allows optimizing the power consumption when the device is
     clocked below the maximum system frequency, to update the voltage scaling value
     regarding system frequency refer to product datasheet.  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /* Enable HSE Oscillator and activate PLL with HSE as source */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 360;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  RCC_OscInitStruct.PLL.PLLR = 6;

  ret = HAL_RCC_OscConfig(&RCC_OscInitStruct);
  if(ret != HAL_OK)
  {
    while(1) { ; }
  }

  /* Activate the OverDrive to reach the 180 MHz Frequency */
  ret = HAL_PWREx_EnableOverDrive();
  if(ret != HAL_OK)
  {
    while(1) { ; }
  }

  /* Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2 clocks dividers */
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  ret = HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5);
  if(ret != HAL_OK)
  {
    while(1) { ; }
  }
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

  /* Infinite loop */
  while (1)
  {
  }
}
#endif

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
