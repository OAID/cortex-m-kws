/**
  ******************************************************************************
  * @file    AID/aid_speech/Src/waveplayer.c 
  * @author  OPEN AI LAB Audio Team
  * @brief   This file provides the Audio Out (playback) interface API
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics International N.V. 
  * All rights reserved.</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without 
  * modification, are permitted, provided that the following conditions are met:
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
#include "waveplayer.h"

/* Private define ------------------------------------------------------------*/


/* Private macro -------------------------------------------------------------*/
/* Private typedef -----------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
static AUDIO_OUT_BufferTypeDef  BufferCtl;
//static int16_t FilePos = 0;
static __IO uint32_t uwVolume = 80;

WAVE_FormatTypeDef WaveFormat;
FIL WavFile;
extern FILELIST_FileTypeDef FileList;
AUDIO_PLAYBACK_StateTypeDef AudioState;
extern void show_on_lcd(char *info);

/* Private function prototypes -----------------------------------------------*/
static AUDIO_ErrorTypeDef GetFileInfo(uint16_t file_idx, WAVE_FormatTypeDef *info);
static uint8_t PlayerInit(uint32_t AudioFreq);
const char * wakeup_file_name = "wakeup.wav";

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Initializes Audio Interface.
  * @param  None
  * @retval Audio error
  */
AUDIO_ErrorTypeDef AUDIO_PLAYER_Init(void)
{
  if(BSP_AUDIO_OUT_Init(OUTPUT_DEVICE_AUTO, uwVolume, I2S_AUDIOFREQ_8K) == 0)
  {
    return AUDIO_ERROR_NONE;
  }
  else
  {
    return AUDIO_ERROR_IO;
  }
}

/**
  * @brief  Starts Audio streaming.    
  * @param  idx: File index
  * @retval Audio error
  */ 
AUDIO_ErrorTypeDef AUDIO_PLAYER_Start(uint8_t idx)
{
  uint32_t bytesread;
  
  f_close(&WavFile);
  if(AUDIO_GetWavObjectNumber() > idx)
  { 
    GetFileInfo(idx, &WaveFormat);
    
    /*Adjust the Audio frequency */
    PlayerInit(WaveFormat.SampleRate); 
    
    BufferCtl.state = BUFFER_OFFSET_NONE;
    
    /* Get Data from USB Flash Disk */
    f_lseek(&WavFile, 0);
    
    /* Fill whole buffer at first time */
    if(f_read(&WavFile, 
              &BufferCtl.buff[0], 
              AUDIO_OUT_BUFFER_SIZE, 
              (void *)&bytesread) == FR_OK)
    {
      AudioState = AUDIO_STATE_PLAY;
//      AUDIO_PlaybackDisplayButtons();
//      BSP_LCD_DisplayStringAt(250, LINE(9), (uint8_t *)"  [PLAY ]", LEFT_MODE);
//      { 
        if(bytesread != 0)
        {
          BSP_AUDIO_OUT_Play((uint16_t*)&BufferCtl.buff[0], AUDIO_OUT_BUFFER_SIZE);
          BufferCtl.fptr = bytesread;
          return AUDIO_ERROR_NONE;
        }
//      }
    }
  }
  return AUDIO_ERROR_IO;
}

/**
  * @brief  Manages Audio process. 
  * @param  None
  * @retval Audio error
  */
AUDIO_ErrorTypeDef _AUDIO_PLAYER_Process(void)
{
  uint32_t bytesread;
  
  if(AudioState == AUDIO_STATE_PLAY)
	{
		while(1)
		{
			if(BufferCtl.fptr >= WaveFormat.FileSize)
			{
				BSP_AUDIO_OUT_Stop(CODEC_PDWN_SW);
				//AudioState = AUDIO_STATE_NEXT;
				return AUDIO_ERROR_NONE;
			}
			
			if(BufferCtl.state == BUFFER_OFFSET_HALF)
			{
				if(f_read(&WavFile, 
									&BufferCtl.buff[0], 
									AUDIO_OUT_BUFFER_SIZE/2, 
									(void *)&bytesread) != FR_OK)
				{ 
					BSP_AUDIO_OUT_Stop(CODEC_PDWN_SW); 
					return AUDIO_ERROR_IO;       
				} 
				BufferCtl.state = BUFFER_OFFSET_NONE;
				BufferCtl.fptr += bytesread; 
			}
			
			if(BufferCtl.state == BUFFER_OFFSET_FULL)
			{
				if(f_read(&WavFile, 
									&BufferCtl.buff[AUDIO_OUT_BUFFER_SIZE /2], 
									AUDIO_OUT_BUFFER_SIZE/2, 
									(void *)&bytesread) != FR_OK)
				{ 
					BSP_AUDIO_OUT_Stop(CODEC_PDWN_SW); 
					return AUDIO_ERROR_IO;       
				} 
	 
				BufferCtl.state = BUFFER_OFFSET_NONE;
				BufferCtl.fptr += bytesread; 
			}	
		}
	}
	else
	{
		return AUDIO_ERROR_IO;
	}

}

/**
  * @brief  Stops Audio streaming.
  * @param  None
  * @retval Audio error
  */
AUDIO_ErrorTypeDef AUDIO_PLAYER_Stop(void)
{
  AudioState = AUDIO_STATE_STOP;
  //FilePos = 0;
  
  BSP_AUDIO_OUT_Stop(CODEC_PDWN_SW);
  f_close(&WavFile);
  return AUDIO_ERROR_NONE;
}

/**
  * @brief  Calculates the remaining file size and new position of the pointer.
  * @param  None
  * @retval None
  */
void BSP_AUDIO_OUT_TransferComplete_CallBack(void)
{
  if(AudioState == AUDIO_STATE_PLAY)
  {
    BufferCtl.state = BUFFER_OFFSET_FULL;
  }
}

/**
  * @brief  Manages the DMA Half Transfer complete interrupt.
  * @param  None
  * @retval None
  */
void BSP_AUDIO_OUT_HalfTransfer_CallBack(void)
{ 
  if(AudioState == AUDIO_STATE_PLAY)
  {
    BufferCtl.state = BUFFER_OFFSET_HALF;
  }
}
/*******************************************************************************
                            Static Functions
*******************************************************************************/

/**
  * @brief  Gets the file info.
  * @param  file_idx: File index
  * @param  info: Pointer to WAV file info
  * @retval Audio error
  */
static AUDIO_ErrorTypeDef GetFileInfo(uint16_t file_idx, WAVE_FormatTypeDef *info)
{
  uint32_t bytesread;
  uint32_t duration;
  uint8_t str[FILEMGR_FILE_NAME_SIZE + 20];  
  
  //if(f_open(&WavFile, (char *)FileList.file[file_idx].name, FA_OPEN_EXISTING | FA_READ) == FR_OK) 
	if(f_open(&WavFile, wakeup_file_name, FA_OPEN_EXISTING | FA_READ) == FR_OK) 
  {
    /* Fill the buffer to Send */
    if(f_read(&WavFile, info, sizeof(WaveFormat), (void *)&bytesread) == FR_OK)
    {
//      sprintf((char *)str, "Playing file (%d/%d): %s", 
//              file_idx + 1, FileList.ptr,
//              (char *)FileList.file[file_idx].name);
			sprintf((char *)str, "Playing file: %s", 
              wakeup_file_name);
      show_on_lcd((char *)str);
 
      sprintf((char *)str,  "Sample rate : %d Hz", (int)(info->SampleRate));
      show_on_lcd((char *)str);
      
      sprintf((char *)str,  "Channels number : %d", info->NbrChannels);  
      show_on_lcd((char *)str);
      
      duration = info->FileSize / info->ByteRate; 
      sprintf((char *)str, "File Size : %d KB [%02d:%02d]", (int)(info->FileSize/1024), (int)(duration/60), (int)(duration%60));
      show_on_lcd((char *)str);
 
      sprintf((char *)str,  "Volume : %d", uwVolume);  
      show_on_lcd((char *)str);
      return AUDIO_ERROR_NONE;
    }
    f_close(&WavFile);
  }
  return AUDIO_ERROR_IO;
}

/**
  * @brief  Initializes the Wave player.
  * @param  AudioFreq: Audio sampling frequency
  * @retval None
  */
static uint8_t PlayerInit(uint32_t AudioFreq)
{ 
  /* Initialize the Audio codec and all related peripherals (I2S, I2C, IOExpander, IOs...) */  
  if(BSP_AUDIO_OUT_Init(OUTPUT_DEVICE_BOTH, uwVolume, AudioFreq) != 0)
  {
    return 1;
  }
  else
  {
    BSP_AUDIO_OUT_SetAudioFrameSlot(CODEC_AUDIOFRAME_SLOT_02);
    return 0;
  } 
}

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
