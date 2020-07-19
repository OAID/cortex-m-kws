/**
  ******************************************************************************
  * @file    AID/aid_speech/Src/command_recognition.c 
  * @author  OPEN AI LAB Audio Team
  * @brief   This file provides the Audio Out (playback) interface API
  ******************************************************************************
  */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "cmsis_os.h"
#include "main.h"
#include "command_recognition.h"
#include "cnn.h"
#include "mfcc.h"
#include "tengine_c_api.h"
#include "tengine_task.h"

#define PREPROCESS_LEN_BYTE (320)
#define WINDOW_SIZE (3)

// threshold for CallBack
int awaken_threshold = 90;

//for record_task
#define MFCC_LEN (NUM_FRAMES * NUM_MFCC_COEFFS)
char pcm_buf[320 * 2];
float mfcc_buf[NUM_MFCC_COEFFS];
float mfcc_buf_test[NUM_MFCC_COEFFS] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

volatile bool run_flag = false;
volatile bool record_stop_flag = false;
volatile bool decode_stop_flag = false;
volatile bool need_pause = false;
volatile bool mfcc_ready = false;

//fifo
typedef struct
{
    char *fifo_buf;                  // = NULL;
    volatile unsigned int read_ptr;  // = 0;
    volatile unsigned int write_ptr; // = 0;
    unsigned int fifo_max_len;       // = 0;
    unsigned int need_len;
    SemaphoreHandle_t fifo_sem;
} FIFI_WITH_SEM;

int show_on_lcd(char *info);
void Fifo_Init(FIFI_WITH_SEM *fifo, int max_len);
void Fifo_Free(FIFI_WITH_SEM *fifo);
int Fifo_Write(FIFI_WITH_SEM *fifo, char *data, int len);
int Fifo_Read(FIFI_WITH_SEM *fifo, char *data, int len, int shift_len);
void Fifo_Reset(FIFI_WITH_SEM *fifo);
int Fifo_Data_Len(FIFI_WITH_SEM *fifo);

AwakenCallback call_back = NULL;
void aid_record_task(void const *argument);
void aid_decode_task(void const *argument);

osThreadId aid_record_thread, aid_decode_thread;
char *aid_record_task_name = "aid_record_thread";
char *aid_decode_task_name = "aid_decode_thread";
static char info[100];

FIFI_WITH_SEM mic_fifo;
#if USE_WEBRTC_AECM
FIFI_WITH_SEM spk_fifo;
#endif
FIFI_WITH_SEM mfcc_fifo;
volatile bool spk_isopen = false;

int AwakenInit(AwakenCallback cb, int threshold, int task_priority)
{
    tprintf("start AwakenInit\n");

    call_back = cb;
    awaken_threshold = threshold;
    Fifo_Init(&mic_fifo, 8192);
#if USE_WEBRTC_AECM
    Fifo_Init(&spk_fifo, 4096);
#endif
    Fifo_Init(&mfcc_fifo, 4096);

    run_flag = true;
    record_stop_flag = false;
    decode_stop_flag = false;

    if (xTaskCreate((TaskFunction_t)aid_record_task, aid_record_task_name, configMINIMAL_STACK_SIZE * 2, NULL, 5, NULL) != pdPASS)
    {
        tprintf("aid_record_thread create error\n");
        return NULL;
    }
    tprintf("aid_record_thread create success\n");

    if (xTaskCreate((TaskFunction_t)aid_decode_task, aid_decode_task_name, configMINIMAL_STACK_SIZE * 4, NULL, 5, NULL) != pdPASS)
    {
        tprintf("aid_record_thread create error\n");
        return NULL;
    }

    tprintf("aid_decode_thread create success\n");
    tprintf("AwakenInit finished\n");

    return 0;
}

// special function to avoid first 2s data.
int AwakenFillBuffer(void)
{
    char temp_buf[NUM_MFCC_COEFFS];
    memset(temp_buf, 0, NUM_MFCC_COEFFS);
    // for quit decode task
    for (int i = 0; i < 98; i++)
    {
        Fifo_Write(&mfcc_fifo, temp_buf, NUM_MFCC_COEFFS);
    }
    return 0;
}

//destroy Awaken library, release some resources
int AwakenDestory(void)
{
    run_flag = 0;

    short junk_data[160];
    AwakenBuffMicData(junk_data, 160);

    while (record_stop_flag == false)
    {
        vTaskDelay(100);
    }
    while (decode_stop_flag == false)
    {
        vTaskDelay(100);
    }

    Fifo_Free(&mic_fifo);
#if USE_WEBRTC_AECM
    Fifo_Free(&spk_fifo);
#endif
    Fifo_Free(&mfcc_fifo);
    tprintf("Awaken destroy done!!!\n");
    return 0;
}

//Mic
int AwakenBuffMicData(short *data, int len)
{
    if (mfcc_ready)
    {
        //show_on_lcd("AwakenBuffMicData!!!\n");
        return Fifo_Write(&mic_fifo, (char *)data, len * 2);
    }
    else
    {
        printf("mfcc_ready = false. data ignored!!!\n");
        return -2;
    }
}

int AwakenBuffSpkData(short *data, int len)
{
#if USE_WEBRTC_AECM
    if (mfcc_ready)
    {
        return Fifo_Write(&spk_fifo, (char *)data, len * 2);
    }
    else
    {
        return -2;
    }

#else
    return 0;
#endif //USE_WEBRTC_AECM
}

int AwakenSpkSwitchNotify(bool is_open, int sample_rate)
{
#if USE_WEBRTC_AECM
    spk_playing = is_open;
    if (spk_playing)
    {
        spk_sample_rate = sample_rate;
    }
    else
    {
        spk_sample_rate = 0;
        Fifo_Reset(&spk_fifo);
    }
#endif //USE_WEBRTC_AECM
    return 0;
}

void PostProcess(int *score)
{
#define MISS_THRESHOLD (2)
    static long wakeup_times = 0;
    static int miss_times = 0;
    static bool wakeup_flag = false;

    for (int j = 1; j < OUT_DIM; j++)
    {
        if ((score[j] > awaken_threshold))
        {
            miss_times = 0;
            if (!wakeup_flag)
            {
                wakeup_flag = true;
                wakeup_times++;

                if (call_back == NULL)
                {
                    show_on_lcd("callback function is NULL\n");
                    return;
                }
                
                (*call_back)(j);
                
                char output[32] = {0};
                switch(j)
                {
                    case 1:
                    {
                        show_on_lcd("Xiaozhi is here.\n");
                        break;
                    }
                    case 2:
                    {
                        show_on_lcd("Dakai Chuanglian.\n");
                        break;
                    }
                    case 3:
                    {
                        show_on_lcd("Guanbi Chuanglian.\n");
                        break;
                    }
                    case 4:
                    {
                        show_on_lcd("Dakai Kongtiao.\n");
                        break;
                    }
                    case 5:
                    {
                        show_on_lcd("Guanbi Kongtiao.\n");
                        break;
                    }
                    case 6:
                    {
                        show_on_lcd("Jiare Moshi.\n");
                        break;
                    }
                    case 7:
                    {
                        show_on_lcd("Zhileng Moshi.\n");
                        break;
                    }
                    case 8:
                    {
                        show_on_lcd("Jiangdi Wendu.\n");
                        break;
                    }
                    case 9:
                    {
                        show_on_lcd("Tiaogao Wendu.\n");
                        break;
                    }
                    case 10:
                    {
                        show_on_lcd("Kaiqi Saofeng.\n");
                        break;
                    }
                    case 11:
                    {
                        show_on_lcd("Qidong Kongtiao.\n");
                        break;
                    }
                    default:
                        sprintf(output, "Score id %d is %d.\n", j, score[j]);
                    show_on_lcd(output);
                }
            }
        }
        else
        {
            miss_times++;
            if ((miss_times > MISS_THRESHOLD * OUT_DIM) && (wakeup_flag))
            {
                wakeup_flag = false;
            }
        }
    }
}

void Fifo_Init(FIFI_WITH_SEM *fifo, int max_len)
{
    fifo->fifo_max_len = max_len;
    fifo->read_ptr = 0;
    fifo->write_ptr = 0;
    fifo->need_len = 0;
    fifo->fifo_buf = calloc(max_len, sizeof(char));
    fifo->fifo_sem = xSemaphoreCreateCounting(3, 0);
}

void Fifo_Free(FIFI_WITH_SEM *fifo)
{
    vSemaphoreDelete(fifo->fifo_sem);
    free(fifo->fifo_buf);
}

int Fifo_Data_Len(FIFI_WITH_SEM *fifo)
{
    return (fifo->write_ptr - fifo->read_ptr);
}

void Fifo_Reset(FIFI_WITH_SEM *fifo)
{
    fifo->read_ptr = 0;
    fifo->write_ptr = 0;
}

int Fifo_Write(FIFI_WITH_SEM *fifo, char *data, int len)
{
    int len1, len2;
    if (Fifo_Data_Len(fifo) + len > fifo->fifo_max_len)
    {
        return -1; //pull whole data to fifo, or discrad it
        len = fifo->fifo_max_len - Fifo_Data_Len(fifo);
    }
    len1 = len + (fifo->write_ptr & (fifo->fifo_max_len - 1));
    if (len1 <= fifo->fifo_max_len)
    {
        len1 = len;
        len2 = 0;
    }
    else
    {
        len1 = fifo->fifo_max_len - (fifo->write_ptr & (fifo->fifo_max_len - 1));
        len2 = len - len1;
    }
    if (len1)
    {
        memcpy((fifo->fifo_buf + (fifo->write_ptr & (fifo->fifo_max_len - 1))), data, len1);
    }
    if (len2)
    {
        memcpy(fifo->fifo_buf, data + len1, len2);
    }
    fifo->write_ptr += len;
    if (Fifo_Data_Len(fifo) >= fifo->need_len)
    {
        xSemaphoreGive(fifo->fifo_sem);
    }
    return 0;
}

int Fifo_Read_Ex(FIFI_WITH_SEM *fifo, char *data, int len, int shift_len, int wait_forever)
{
    int len1, len2;
    fifo->need_len = len;
    while (len > Fifo_Data_Len(fifo))
    {
        int ret = xSemaphoreTake(fifo->fifo_sem, wait_forever);

        if (ret != pdTRUE)
        {   //timeout
            if (wait_forever != portMAX_DELAY)
                return -1;
        }
    }
    len1 = len + (fifo->read_ptr & (fifo->fifo_max_len - 1));
    if (len1 <= fifo->fifo_max_len)
    {
        len1 = len;
        len2 = 0;
    }
    else
    {
        len1 = fifo->fifo_max_len - (fifo->read_ptr & (fifo->fifo_max_len - 1));
        len2 = len - len1;
    }
    if (len1)
    {
        memcpy(data, fifo->fifo_buf + (fifo->read_ptr & (fifo->fifo_max_len - 1)), len1);
    }
    if (len2)
    {
        memcpy(data + len1, fifo->fifo_buf, len2);
    }
    fifo->read_ptr += shift_len;
    return 0;
}

//Read len byte data but only add read_ptr for shift_len
//some data may be readed more then one time
int Fifo_Read(FIFI_WITH_SEM *fifo, char *data, int len, int shift_len)
{
    return Fifo_Read_Ex(fifo, data, len, shift_len, portMAX_DELAY);
}

int Fifo_Read_Nosem(FIFI_WITH_SEM *fifo, char *data, int len)
{
    return Fifo_Read_Ex(fifo, data, len, len, 5); //2ms
}

static signed char convert_mfcc_to_char(float mfcc_f)
{
	float feature_tmp = mfcc_f;
	feature_tmp = round(feature_tmp);
	if(feature_tmp > 127)
	{
		feature_tmp = 127;
	}
	if(feature_tmp < -128)
	{
		feature_tmp = -128;
	}
	//convert to char
	return feature_tmp; 
}

int inputdata_preprocess(float* in_data, q7_t* out_data)
{
	char input_data_count = 0 ;
	for(int i=0; i<CONV_DATA_LEN; i++)   
	{

		for(int j=0; j<INPUT_FEATURE_DIM_W; j++)
		{
			if(j==0)
			{
				out_data[input_data_count*INPUT_FEATURE_DIM_W+j] = convert_mfcc_to_char(in_data[i*NUM_MFCC_COEFFS+j]);
			}
			else
			{
				out_data[input_data_count*INPUT_FEATURE_DIM_W+j] = convert_mfcc_to_char(in_data[i*NUM_MFCC_COEFFS+j]*2);
			}
			
		}
		input_data_count++;

	}
	return 0 ;
}

void aid_record_task(void const *argument)
{
    MFCC_init();
    int pcm_head = 0;

    mfcc_ready = true;
    while (run_flag)
    {
        Fifo_Read(&mic_fifo, pcm_buf + pcm_head, PREPROCESS_LEN_BYTE, PREPROCESS_LEN_BYTE);

        pcm_head += PREPROCESS_LEN_BYTE;
        if (MFCC_FRAME_LEN * 2 <= pcm_head)
        {
            unsigned long start_time = xTaskGetTickCount();
            MFCC_mfcc_compute((int16_t *)pcm_buf, mfcc_buf);

            memmove(pcm_buf, pcm_buf + MFCC_FRAME_SHIFT * 2, pcm_head - MFCC_FRAME_SHIFT * 2);
            pcm_head -= MFCC_FRAME_SHIFT * 2;

            Fifo_Write(&mfcc_fifo, (char *)mfcc_buf, NUM_MFCC_COEFFS * sizeof(float));
        }
    }

    show_on_lcd("record_task quit!\n");
record_quit:
#if USE_WEBRTC_AGC_NS
    WebRtcNsx_Free(nsx_handle);
    WebRtcAgc_Free(agc_handle);
#endif

    run_flag = false;
    memset(mfcc_buf, 0, NUM_MFCC_COEFFS);
    // for quit decode task
    for (int i = 0; i < 99; i++)
    {
        Fifo_Write(&mfcc_fifo, (char *)mfcc_buf, NUM_MFCC_COEFFS * sizeof(float));
    }
    MFCC_delete();
    record_stop_flag = true;
    show_on_lcd("record_task stop!\n");
    vTaskDelete(aid_record_thread);
}

void aid_decode_task(void const *argument)
{
    graph_t graph = NULL;
    int smoothed_score[OUT_DIM] = {0};
    q7_t output_buf[WINDOW_SIZE][OUT_DIM] = {0};
    float mfcc_data[NUM_MFCC_COEFFS * CONV_DATA_LEN] = {0};
    int output_write_ptr = 1;

    /* tengien lite initial, and load graph */
    graph = tengine_lite_init(graph);

    /* set point of input data */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int input_size = get_tensor_buffer_size(input_tensor);
    char *input_buf = (char *)malloc(input_size * sizeof(char));
    if (set_tensor_buffer(input_tensor, (void *)input_buf, input_size) < 0)
    {
        printf("set input tensor buffer failed\n");
        goto TENGINE_ERR;
    }

    /* set point of output data */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    char *output = get_tensor_buffer(output_tensor);

    while (run_flag)
    {
        /* get input audio data */
        Fifo_Read(&mfcc_fifo, (char *)mfcc_data, NUM_MFCC_COEFFS * CONV_DATA_LEN * sizeof(float), NUM_MFCC_COEFFS * CONV_DATA_LEN * sizeof(float));

        /* preprocess input data */
        inputdata_preprocess(mfcc_data, (q7_t *)input_buf);

        /* nn inference */
        run_graph(graph, 1);

        /* process result */
        for (int i = 0; i < OUT_DIM; i++)
        {
            output_buf[output_write_ptr][i] = output[i];
        }

        output_write_ptr = (output_write_ptr + 1) % WINDOW_SIZE;
        for (int i = 0; i < OUT_DIM; i++)
        {
            for (int j = 0; j < WINDOW_SIZE; j++)
            {
                smoothed_score[i] += output_buf[j][i];
            }
            smoothed_score[i] /= WINDOW_SIZE; //(WINDOW_SIZE + 1); //WINDOW_SIZE
        }

        PostProcess(smoothed_score);
    }

TENGINE_ERR:
    tengine_lite_release(graph);
    run_flag = false;
    free(input_buf);

    printf("aid_decode_thread quit!\n");
    decode_stop_flag = true;
    vTaskDelete(aid_decode_thread);
}
