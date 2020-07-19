#ifndef _COMMAND_RECOGNITION_H_
#define _COMMAND_RECOGNITION_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>



#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


typedef void (*AwakenCallback)(int);

//init Awaken library, it will run automaticly, call the cb when detect awaken word
//cb: callback function, will be called when awaken word detected
//threshold:  [default] 90, set the threshold of awaken word, max value: 127
//task_priority: set task priority of Awaken task
// return 0 £º init success; other : init failed
int AwakenInit(AwakenCallback cb, int threshold, int task_priority);


// special function to avoid first 2s data.
int AwakenFillBuffer(void);

//destroy Awaken library, release some resources
// return 0 £º destory success; other : destory failed
int AwakenDestory(void);

//input data from Microphone
//data: point to input data
//len:  data length(unit: short)
// return 0 £º success; other : failed
int AwakenBuffMicData(short *data, int len);

//input data from Speaker for AEC
//data: point to input data
//len:  data length(unit: short)
// return 0 £º success; other : failed
int AwakenBuffSpkData(short *data, int len);

//notify state of Speaker
//is_open: ture, open Speaker
//         false, close Speaker
//sample_rate: sample rate of speaker data
// return 0 £º success; other : failed
int AwakenSpkSwitchNotify(bool is_open, int sample_rate);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif//__AUDIO_DISPLAY_H_
