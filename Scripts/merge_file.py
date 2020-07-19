# -*- coding: utf-8 -*

import os
#获取目标文件夹的路径
filename1 = './cnn_weights.h'
filename2 = './cnn.h'
#获取当前文件夹中的文件名称列表  
#打开当前目录下的result.json文件，如果没有则创建
f=open('tiny_param_generated.h','w')
f.write("/*\n")
f.write(" * This header file is generated automatically, please do not modify.\n")
f.write(" */\n\n")
f.write("#ifndef __TINY_GRAPH_GENERATED_H__\n#define __TINY_GRAPH_GENERATED_H__\n\n")
#先遍历文件名
#遍历单个文件，读取行数

for line in open(filename1):
    f.writelines(line)
    f.write('\n')
#关闭文件
f.write("\n\n");
f.write("static const signed char conv_0_weight_data[] = FIRST_CONV_W_0;\n")
f.write("static const signed char conv_0_bias_data[] = FIRST_CONV_B_0;\n")
f.write("\n")
f.write("static const signed char conv_1_weight_data[] = SECOND_CONV_W_0;\n")
f.write("static const signed char conv_1_bias_data[] = SECOND_CONV_B_0;\n")
f.write("\n")
f.write("static const signed char conv_2_weight_data[] = THIRD_CONV_W_0;\n")
f.write("static const signed char conv_2_bias_data[] = THIRD_CONV_B_0;\n")
f.write("\n")
f.write("static const signed char conv_3_weight_data[] = FOURTH_CONV_W_0;\n")
f.write("static const signed char conv_3_bias_data[] = FOURTH_CONV_B_0;\n")
f.write("\n")
f.write("static const signed char fc_4_weight_data[] = LINEAR_W_0;\n")
f.write("static const signed char fc_4_bias_data[] = LINEAR_B_0;\n")
f.write("\n")
f.write("static const signed char fc_5_weight_data[] = FIRST_FC_W_0;\n")
f.write("static const signed char fc_5_bias_data[] = FIRST_FC_B_0;\n")

f.write("\n")
f.write("static const signed char fc_7_weight_data[] = FINAL_FC_W_0;\n")
f.write("static const signed char fc_7_bias_data[] = FINAL_FC_B_0;\n")

f.write("\n\n#endif\n\n");
f.close()
