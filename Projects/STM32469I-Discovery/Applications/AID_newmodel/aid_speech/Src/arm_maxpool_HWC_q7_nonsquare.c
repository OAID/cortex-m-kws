#include "arm_math.h"
#include "arm_nnfunctions.h"

#if defined (ARM_MATH_DSP)

/**
 * @brief A few utility functions used by pooling functions
 *
 * 
 */
static void compare_and_replace_if_larger_q7(q7_t * base,   // base data
                                             q7_t * target, // compare target
                                             const uint16_t length  // data length
)
{
    q7_t     *pIn = base;
    q7_t     *pCom = target;
    union arm_nnword in;
    union arm_nnword com;
    uint16_t  cnt = length >> 2;

    while (cnt > 0u)
    {
        in.word = *__SIMD32(pIn);
        com.word = *__SIMD32(pCom)++;

        // if version
        if (com.bytes[0] > in.bytes[0])
            in.bytes[0] = com.bytes[0];
        if (com.bytes[1] > in.bytes[1])
            in.bytes[1] = com.bytes[1];
        if (com.bytes[2] > in.bytes[2])
            in.bytes[2] = com.bytes[2];
        if (com.bytes[3] > in.bytes[3])
            in.bytes[3] = com.bytes[3];

        *__SIMD32(pIn)++ = in.word;

        cnt--;
    }
}

#endif                          // ARM_MATH_DSP


void
arm_maxpool_HWC_q7_nonsquare(q7_t * Im_in,
                   const uint16_t dim_im_in_x,
				   const uint16_t dim_im_in_y,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel,
                   const uint16_t padding,
                   const uint16_t stride, 
				   const uint16_t dim_im_out_x, 
				   const uint16_t dim_im_out_y, q7_t * bufferA, q7_t * Im_out)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_x, i_y;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < dim_im_in_y; i_y++)
    {

        for (i_x = 0; i_x < dim_im_out_x; i_x++)
        {
            /* for each output pixel */
            q7_t     *target = Im_in + (i_y * dim_im_in_x + i_x) * ch_im_in;
            q7_t     *win_start;
            q7_t     *win_stop;
            if (i_x * stride - padding < 0)
            {
                win_start = target;
            } else
            {
                win_start = Im_in + (i_y * dim_im_in_x + i_x * stride - padding) * ch_im_in;
            }

            if (i_x * stride - padding + dim_kernel >= dim_im_in_x)
            {
                win_stop = Im_in + (i_y * dim_im_in_x + dim_im_in_x) * ch_im_in;
            } else
            {
                win_stop = Im_in + (i_y * dim_im_in_x + i_x * stride - padding + dim_kernel) * ch_im_in;
            }

            /* first step is to copy over initial data */
            /* arm_copy_q7(win_start, target, ch_im_in); */
            memmove(target, win_start, ch_im_in);

            /* start the max operation from the second part */
            win_start += ch_im_in;
            for (; win_start < win_stop; win_start += ch_im_in)
            {
                compare_and_replace_if_larger_q7(target, win_start, ch_im_in);
            }
        }
    }

    /* then does the pooling along y axis */
    for (i_y = 0; i_y < dim_im_out_y; i_y++)
    {

        /* for each output row */
        q7_t     *target = Im_out + i_y * dim_im_out_x * ch_im_in;
        q7_t     *row_start;
        q7_t     *row_end;
        /* setting the starting row */
        if (i_y * stride - padding < 0)
        {
            row_start = Im_in;
        } else
        {
            row_start = Im_in + (i_y * stride - padding) * dim_im_in_x * ch_im_in;
        }
        /* setting the stopping row */
        if (i_y * stride - padding + dim_kernel >= dim_im_in_y)
        {
            row_end = Im_in + dim_im_in_x * dim_im_in_y * ch_im_in;
        } else
        {
            row_end = Im_in + (i_y * stride - padding + dim_kernel) * dim_im_in_x * ch_im_in;
        }

        /* copy over the first row */
        /* arm_copy_q7(row_start, target, dim_im_out * ch_im_in); */
        memmove(target, row_start, dim_im_out_x * ch_im_in);

        /* move over to next row */
        row_start += ch_im_in * dim_im_in_x;

        for (; row_start < row_end; row_start += dim_im_in_x * ch_im_in)
        {
            compare_and_replace_if_larger_q7(target, row_start, dim_im_out_x * ch_im_in);
        }
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int       max = -129;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }

#endif                          /* ARM_MATH_DSP */

}
