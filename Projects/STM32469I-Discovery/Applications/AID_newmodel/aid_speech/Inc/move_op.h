#ifndef _MOVE_OP_H_
#define _MOVE_OP_H_

#include <string.h>
#include <stdio.h>

typedef struct memmove {
	int flag ;
	int fillin_buffer_size ;
	int current_buffer_size ;
	int buffer_out_size;
	int datain_cnt ; 
	int start_move_addr ; 
	int move_size ;
	signed char *buffer;
}MV ;

int mem_move_op(signed char *buffer_last , \
								const signed char *new_buffer , \
								signed char *buffer_out , \
								int buffer_out_size , \
								int offset , \
								int keep);
	

int first_move_op(const signed char *new_buffer , \
								signed char *buffer_out , \
								MV* mv_para);			

int second_move_op(const signed char *new_buffer , \
								signed char *buffer_out , \
								MV* mv_para);	
								
int third_move_op(const signed char *new_buffer , \
								signed char *buffer_out , \
								MV* mv_para);	
								
int fourth_move_op(const signed char *new_buffer , \
								signed char *buffer_out , \
								MV* mv_para);	

int fifth_move_op(const signed char *new_buffer , signed char *buffer_out , MV* mv_para)		;		

int move_op(const signed char *new_buffer , \
								signed char *buffer_out , \
								MV* mv_para);	
								
								
#endif