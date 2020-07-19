/**
  ******************************************************************************
  * @file    AID/aid_speech/Src/test.c 
  * @author  OPEN AI LAB Audio Team
  * @brief   This file provides the Audio Out (playback) interface API
  ******************************************************************************
  */
  
#include "main.h"
#include "command_recognition.h"
#include "cmsis_os.h"
#include "waveplayer.h"
#include "waverecorder.h"

FATFS USBH_FatFs;
USBH_HandleTypeDef hUSBHost;
AUDIO_ApplicationTypeDef AppliState = APPLICATION_IDLE;
TS_StateTypeDef  TS_State = {0};
static void USBH_UserProcess(USBH_HandleTypeDef *phost, uint8_t id);
static void AUDIO_InitApplication(void);

extern FILELIST_FileTypeDef FileList;

extern uint8_t _pHeaderBuff;
extern WAVE_FormatTypeDef _WaveFormat;
extern AUDIO_IN_BufferTypeDef  _BufferCtl;

void show_on_lcd(char *info);
static void clear_screen();

//static bool wakeup_flag = false;
osThreadId mic_in_handle;
static void hardware_record_task(void const *argument);

void stereo_to_mono(const uint16_t* src_audio,int samples_per_channel,uint16_t* dst_audio) 
{
    for (int i = 0; i < samples_per_channel; i++) 
    {
        dst_audio[i] = src_audio[2 * i];
    }
}

void stereo_to_mono_with_resample_16_to_8(const uint16_t* src_audio,int samples_per_channel,uint16_t* dst_audio) 
{
    for (int i = 0; i < samples_per_channel/2; i++) 
    {
        dst_audio[i] = src_audio[4 * i];
    }
} 

static void hardware_record_task(void const *argument)
{
  show_on_lcd("Enter Hardware_record_task......\n");

  uint16_t *one_channel =(uint16_t *)malloc(AUDIO_IN_PCM_BUFFER_SIZE/8 * sizeof(uint16_t));

  /* Configure the audio recorder: sampling frequency, bits-depth, number of channels */
  if(_AUDIO_REC_Start() == AUDIO_ERROR_NONE)
  {
    //while(_BufferCtl.fptr<1280000)
    while(1)
    {
      /* Check if there are Data to write to USB Key */
      if(_BufferCtl.wr_state == BUFFER_FULL)
      {			
        stereo_to_mono_with_resample_16_to_8(_BufferCtl.pcm_buff + _BufferCtl.offset, AUDIO_IN_PCM_BUFFER_SIZE/4, one_channel);
        AwakenBuffMicData((short *)one_channel, AUDIO_IN_PCM_BUFFER_SIZE/8);
        //_BufferCtl.fptr += byteswritten;
        _BufferCtl.wr_state =  BUFFER_EMPTY;
      }
      osDelay(1);
    }
    /* Stop recorder */
    BSP_AUDIO_IN_Stop();
    HAL_Delay(150);
  }
}

void play_wakeup()
{
  /* Init Audio interface */
  AUDIO_PLAYER_Init();
  AUDIO_PLAYER_Start(0);

  AUDIO_ErrorTypeDef ret = _AUDIO_PLAYER_Process();
  if(ret == AUDIO_ERROR_IO)
  {
    show_on_lcd("play error.\n");
  }

  AUDIO_PLAYER_Stop();
}

/* callback function for awaken lib */
void test_cb(int arg)
{
    show_on_lcd("\n=======================\n");

    static long wakeup_times = 0;
    char output[20];

    wakeup_times++;
    sprintf(output, "Wakeup: %ld\n", wakeup_times);
    show_on_lcd(output);
  
    /* play wakeup music. */
    show_on_lcd("Xiaozhi is coming!\n");
}

/* clear all scoen */
static int line = 0;
static void clear_screen()
{
  BSP_LCD_Clear(LCD_COLOR_BLACK);
  line=0;
}

void show_on_lcd(char *info)
{
  if(line > YWINDOW_SIZE)
  {
    clear_screen();
  }
  BSP_LCD_DisplayStringAtLine(line, (uint8_t *)info);
  line++;
}

/*******************************************************************************
                            Static Function
*******************************************************************************/

/**
  * @brief  Audio Application Init.
  * @param  None
  * @retval None
  */
static void AUDIO_InitApplication(void)
{
  uint8_t  lcd_status = LCD_OK;
  //uint32_t ts_status  = TS_OK;

  /* Configure User Button */
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);              
  
  /* Configure LED1 */
  BSP_LED_Init(LED1);
  
  /* Initialize the SDRAM */
  if (BSP_SDRAM_Init() != SDRAM_OK)
  {
    /* User can add here some code to deal with this error */
    while (1)
    {
    }
  }

  /*##-1- LCD DSI initialization ########################################*/

  /* Initialize and start the LCD display in mode 'lcd_mode'
   *  Using LCD_FB_START_ADDRESS as frame buffer displayed contents.
   *  This buffer is modified by the BSP (draw fonts, objects depending on BSP calls).
   */

  /* Set Portrait orientation if needed, by default orientation is set to
     Landscape */
  
  /* Initialize DSI LCD */
  //  BSP_LCD_InitEx(LCD_ORIENTATION_PORTRAIT); /* uncomment if Portrait orientation is needed */
  BSP_LCD_Init(); /* Uncomment if default config (landscape orientation) is needed */
  while(lcd_status != LCD_OK);

  BSP_LCD_LayerDefaultInit(0, LCD_FB_START_ADDRESS);   
  BSP_LCD_SelectLayer(0);

  /*##-2- Touch screen initialization ########################################*/
  BSP_TS_ResetTouchData(&TS_State);

  
  /* Init the LCD Log module */
  LCD_LOG_Init();
  
  LCD_LOG_SetHeader((uint8_t *)"Key Word Spooting by OPEN AI LAB");
  //show_on_lcd("Key Word Spooting by OPEN AI LAB")
  show_on_lcd("USB Host library started.\n"); 
  
  /* Start Audio interface */
  show_on_lcd("Starting Audio Demo");
}

/**
  * @brief  User Process
  * @param  phost: Host Handle
  * @param  id: Host Library user message ID
  * @retval None
  */
static void USBH_UserProcess(USBH_HandleTypeDef *phost, uint8_t id)
{
  switch(id)
  { 
  case HOST_USER_SELECT_CONFIGURATION:
    break;
    
  case HOST_USER_DISCONNECTION:
    if(f_mount(NULL, "", 0) != FR_OK)
    {
      LCD_ErrLog("ERROR : Cannot DeInitialize FatFs! \n");
    }
    AppliState = APPLICATION_DISCONNECT;
    break;

  case HOST_USER_CLASS_ACTIVE:
    AppliState = APPLICATION_READY;
    break;
 
  case HOST_USER_CONNECTION:
    /* Link the USB Mass Storage disk I/O driver */
    if(FATFS_LinkDriver(&USBH_Driver, (char*)"0:/") != 0)
    {
    }
    if(f_mount(&USBH_FatFs, "", 0) != FR_OK)
    {  
      LCD_ErrLog("ERROR : Cannot Initialize FatFs! \n");
    }
    AppliState = APPLICATION_START;
    break;
   
  default:
    break;
  }
}

int task_init(void)
{
  AUDIO_InitApplication();
  printf("task_init...\n");
  printf("%s %s : Let's go \n", __DATE__, __TIME__);
  
  AwakenInit(test_cb, 84, 4);

  if (xTaskCreate((TaskFunction_t)hardware_record_task, "MIC Record", configMINIMAL_STACK_SIZE*2, NULL, 3, NULL) != pdPASS)
  {
    printf("mic_in_handle create error\n");
    return NULL;
  }

  printf("mic_in_handle create success\n");
  
  clear_screen();
  
  /* Start scheduler */
  vTaskStartScheduler();

  /* We should never get here as control is now taken by the scheduler */
  for(;;);

  return 0;
}

void vApplicationMallocFailedHook( void )
{
  static int ttttt = 0;
  while(1)
  {
      ttttt++;
  }
}

void vApplicationStackOverflowHook( TaskHandle_t xTask, char *pcTaskName )
{
  static int ttttt = 0;
  while(1)
  {
    ttttt++;
  }
}
