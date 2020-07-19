#ifndef _AID_SIMPLE_RESAMPLER_H_
#define _AID_SIMPLE_RESAMPLER_H_


typedef short int16_t;
typedef int int32_t;
typedef unsigned int uint32_t;


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

//cal out_len after resample.
//in_sample_Rate: input data sample rate
//out_sample_Rate: output data sample rate
//in_len: input data length (real in_buf size)
// return : out_len
uint32_t get_out_len(int32_t in_sample_Rate, int32_t out_sample_Rate,uint32_t in_len);

//resampleData
//in_buf: point to input data
//in_sample_Rate: input data sample rate
//in_len: input data length (real in_buf size)
//out_buf: point to output data after resample
//out_sample_Rate: output data sample rate
//out_len: output data length (get_out_len() result)
// return 0 ： success;
int resampleData(const int16_t *in_buf, int32_t in_sample_Rate, uint32_t in_len, int16_t *out_buf,
                  int32_t out_sample_Rate, uint32_t out_len);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif//_AID_SIMPLE_RESAMPLER_H_