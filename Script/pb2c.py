import tensorflow as tf
import argparse
import os
import sys
import datetime
import re
import numpy as np

from tensorflow.python.framework import tensor_util

graph_name = "tengine-lite"

full_path=os.getcwd()
print(full_path)
out_path = full_path+"/out";
print(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

def header_file_comment_info(header_file):
    header_file.write(" /*\n")
    header_file.write(" * Licensed to the Apache Software Foundation (ASF) under one\n")
    header_file.write(" * or more contributor license agreements.  See the NOTICE file\n")
    header_file.write(" * distributed with this work for additional information\n")
    header_file.write(" * regarding copyright ownership.  The ASF licenses this file\n")
    header_file.write(" * to you under the Apache License, Version 2.0 (the\n")
    header_file.write(" * License); you may not use this file except in compliance\n")
    header_file.write(" * with the License.  You may obtain a copy of the License at\n")
    header_file.write(" *\n")
    header_file.write(" *   http://www.apache.org/licenses/LICENSE-2.0\n")
    header_file.write(" *\n")
    header_file.write(" * Unless required by applicable law or agreed to in writing,\n")
    header_file.write(" * software distributed under the License is distributed on an\n")
    header_file.write(" * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n")
    header_file.write(" * KIND, either express or implied.  See the License for the\n")
    header_file.write(" * specific language governing permissions and limitations\n")
    header_file.write(" * under the License.\n")
    header_file.write(" */\n")
    header_file.write("\n")
    header_file.write("/*\n")
    header_file.write(" * Copyright (c) 2019, OPEN AI LAB\n")
    header_file.write(" * Author: python scripts\n")
    header_file.write(" */\n")
    header_file.write(" \n")
    header_file.write(" /*\n")
    header_file.write(" * This header file is generated automatically, please do not modify.\n")
    header_file.write(" */\n")
    header_file.write(" \n")
    header_file.write(" \n")
    header_file.write("\n")
    header_file.write("#ifndef __TINY_PARAM_GENERATED_H__\n")
    header_file.write("#define __TINY_PARAM_GENERATED_H__\n")

def C_code_file_comment_info(c_file):
    c_file.write("/*\n")
    c_file.write(" * Licensed to the Apache Software Foundation (ASF) under one\n")
    c_file.write(" * or more contributor license agreements.  See the NOTICE file\n")
    c_file.write(" * distributed with this work for additional information\n")
    c_file.write(" * regarding copyright ownership.  The ASF licenses this file\n")
    c_file.write(" * to you under the Apache License, Version 2.0 (the\n")
    c_file.write(" * License); you may not use this file except in compliance\n")
    c_file.write(" * with the License.  You may obtain a copy of the License at\n")
    c_file.write(" *\n")
    c_file.write(" *   http://www.apache.org/licenses/LICENSE-2.0\n")
    c_file.write(" *\n")
    c_file.write(" * Unless required by applicable law or agreed to in writing,\n")
    c_file.write(" * software distributed under the License is distributed on an\n")
    c_file.write(" * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n")
    c_file.write(" * KIND, either express or implied.  See the License for the\n")
    c_file.write(" * specific language governing permissions and limitations\n")
    c_file.write(" * under the License.\n")
    c_file.write(" */\n\n")

    c_file.write("/*\n")
    c_file.write(" * Copyright (c) 2019, OPEN AI LAB\n")
    c_file.write(" * Author: python scripts\n")
    c_file.write(" */\n\n")
    c_file.write(" /*\n")
    c_file.write(" * This header file is generated automatically, please do not modify.\n")
    c_file.write(" */\n")
    c_file.write("#include <stdio.h>\n\n")
    c_file.write("#include \"tiny_graph.h\"\n")
    c_file.write("#include \"tiny_param_generated.h\"\n\n")


def get_shift_value(graph ,  write_file ):
        dec_bits_output = [] 
        dec_bit_weiths = []
        dec_bit_name = []
        shift_node_name = []
        
        for node in graph.node:
            if 'FakeQuantWithMinMaxVars' in node.name :
                continue;
            if 'Reshape' in node.name :
                continue;
            if node.op == 'Const' or ('Conv2D' in node.op) or ('MatMul' in node.op):            
                shift_node_name.append(node.name)

        for node in graph.node:
            print node.name
            print node.op
            if 'FakeQuantWithMinMaxVars' in node.name and node.op == 'Const':
                text = tensor_util.MakeNdarray(node.attr['value'].tensor)
                value = int(np.ceil(np.log2(abs(text))))
                value = 7 - value 
                dec_bits_output.append(value)
            if 'FakeQuantWithMinMaxVars' in node.name or 'Reshape' in node.name  :
                continue;
            if node.op == 'Const':
                 text = tensor_util.MakeNdarray(node.attr['value'].tensor)
                 text = text.flatten()  
                 min_value = text.min()
                 max_value = text.max()
                 int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
                 dec_bits = 7 - int_bits
                 dec_bit_weiths.append(dec_bits)
                 dec_bit_name.append(str(node.name))
        print dec_bits_output
        print dec_bit_weiths
        print dec_bit_name
        length = len(dec_bit_weiths)/2 
        for i in range(length):
          tmp_dec = dec_bits_output[2*i] + dec_bit_weiths[2*i]
          bias_lshift = tmp_dec - dec_bit_weiths[2*i+1]
          output_rshift = tmp_dec - dec_bits_output[2*(i+1)]
          print bias_lshift
          print output_rshift
          write_file.write("#define  %s_SHIFT  0\n"%str(shift_node_name[3*i+0]).upper())
          write_file.write("#define  %s_SHIFT  %d\n"% (str(shift_node_name[3*i+1].upper()), bias_lshift))
          write_file.write("#define  %s_SHIFT  %d\n"% (str(shift_node_name[3*i+2].upper()), output_rshift ))
          print("#define  %s_SHIFT  \n"%str(shift_node_name[3*i+0]).upper())
         
        
        
                


def get_ops_from_pb_and_write_h(graph):
    with open('tiny_param_generated.h','w+') as wh:
        header_file_comment_info(wh)

        print("Tensor name : ")
        tensor_name_list = [tensor.name for tensor in graph.node]
#        for tensor_name in tensor_name_list:
#            print(tensor_name)
        
        print("\nNode name : ")
        for node in graph.node:
            if 'FakeQuantWithMinMaxVars' in node.name :
                continue;
            if node.op == 'Const' :
                 text = tensor_util.MakeNdarray(node.attr['value'].tensor)
                 text = text.flatten()  
                 #print text  
                 min_value = text.min()
                 max_value = text.max()
                 print ( min_value)
                 print ( max_value)
                 int_bits = int(np.ceil(np.log2(max(abs(min_value),abs(max_value)))))
                 dec_bits = 7 - int_bits
                 # convert to [-128,128) or int8
                 text = np.round(text*2**dec_bits)     
                 name = node.name.replace('/','_')
                 wh.write("\n#define %s_DATA {" % name.upper())
#                 with open('model.h','ab') as ff:
#                    np.savetxt(ff,text,fmt='%d',delimiter=', ',newline=', ')
                 for idx,value in enumerate(text):   
#                    if idx%20==0:
#                        wh.write("\\ \r") 
                    wh.write("%d ,"% value) 
                
                 wh.seek(-1,1)
                 wh.write("}\n")
                 print ("node : %s shit bit = %d  dec_bits = %d " % (node.name ,int_bits, dec_bits))
        wh.write("\n")
        
        get_shift_value(graph,wh)

        for node in graph.node:
            print(node.name)
            if 'FakeQuantWithMinMaxVars' in node.name :
                continue;
            if 'Reshape' in node.name :
                continue;
            name = node.name.replace('/','_')
            if node.op == 'Const':
                wh.write("\nstatic const signed char %s_data[] = %s_DATA;\n"% (name ,name.upper()))
    
        wh.write("\n")
        wh.write("\n")
        wh.write("#endif\n")

def get_ops_from_pb(graph , save_ori_network=True):
    if save_ori_network:
#        with open('ori_network.txt','w+') as w: 
        with open('tiny_graph_generated.c','w+') as w: 
            OPS=graph.get_operations()
            C_code_file_comment_info(w)
            for op in OPS:
                    print("OP:  %s   type=%s" % (op.name , op.type))
                    if op.type == 'Conv2D' : 
                        print("          attr=%s"% op.get_attr("strides"))
                        print("          attr=%s"% op.get_attr("dilations"))
                        print("          format=%s"% op.get_attr("data_format"))
                    if op.type == 'MaxPool' : 
                        print("          attr=%s"% op.get_attr("strides"))
                        print("          attr=%s"% op.get_attr("ksize"))
                        for a in op.get_attr("ksize"):
                            print("               %d"% a)
                    print("         Input tenseor : total-%d" % len(op.inputs)) ##input num 
                    for input_tensor in op.inputs:
                        print("                Name :%s "% input_tensor.name)
                        if op.type == "DecodeWav":
                            print("                Shape Size :unkown ")
                        else   : 
                            print("                Shape Size :%s "% len(input_tensor.shape))
                        print("                Shape:%s "% input_tensor.shape)
                        print("                Datye:%s "% str([input_tensor.dtype]))
                        print("                *****************************")
                        print("                 ")    
                    print("         Output tenseor : total-%d" % len(op.outputs)) ##output num ") 
                    for tensor in op.outputs:
                        print("                 Name:%s" % tensor.name)    
                        print("                 dtype:%s" % str([tensor.dtype]))
############################# tensor structure#############################################
            for op in OPS:
                if op.type != 'Const' and op.type != 'Identity' and op.type != 'FakeQuantWithMinMaxVars' :
                    for idx ,input_tensor in enumerate(op.inputs) :
                        if 'FakeQuantWithMinMaxVars' in input_tensor.name :
                            continue ;
                        w.write("static const struct tiny_tensor %s ={\n" %str(input_tensor.name).replace(':','_').replace('/read','').replace('/','_') )
                        w.write("           .dim_num = %d,\n" % len(input_tensor.shape))
                        w.write("           .dims = { ")
                        for dim_value in input_tensor.shape.as_list() :
                            if dim_value == None :
                                w.write("1,")
                            else :
                                w.write("%s," % dim_value)
                        w.seek(-1,1)
                        w.write("},\n")
                        # shift value 
                        if op.type == 'Relu' or op.type == 'MaxPool' or op.type == 'Reshape' or ('Reshape' in input_tensor.name) or ('MaxPool' in input_tensor.name) or ('Relu' in input_tensor.name):
                            w.write("           .shift = 0,\n")
                        else :
                            w.write("           .shift = %s_SHIFT,\n"%str(input_tensor.name).replace(':','_').replace('/read','').replace('/','_').upper()[:-2])
                        w.write("           .data_type = NN_DT_Q7,\n")
                        if idx == 0:
                            if input_tensor.name != 'Placeholder:0':
                                w.write("           .tensor_type = NN_TENSOR_VAR,\n")
                            else: 
                                w.write("           .tensor_type = NN_TENSOR_INPUT,\n")
                            w.write("           .data = NULL,\n" )
                        else :
                            w.write("           .tensor_type = NN_TENSOR_CONST,\n")
                            w.write("           .data = %s_data,\n" %str(input_tensor.name).replace(':','_').replace('/read','').replace('/','_')[:-2]  )
                        w.write("};\n\n")
                 
            output_tensors = OPS[-1].outputs
            for output_tensor in output_tensors:
                 w.write("static const struct tiny_tensor %s ={\n" %str(output_tensor.name).replace(':','_').replace('/','_') )
                 w.write("           .dim_num = %d\n" % len(output_tensor.shape))
                 w.write("           .dims = { ")
                 for dim_value in output_tensor.shape.as_list() :
                     if dim_value == None :
                        w.write("1,")
                     else :
                        w.write("%s," % dim_value)
                 w.seek(-1,1)
                 w.write("},\n")
                 w.write("           .shift = 0,\n")
                 w.write("           .data_type = NN_DT_Q7,\n" )
                 w.write("           .tensor_type = NN_TENSOR_VAR,\n")
                 w.write("           .data = NULL,\n")
                 w.write("};\n\n")

############################# param structure#############################################
            for op in OPS:
                if op.type == 'MaxPool' or op.type == 'AvgPool':
                    w.write("static const struct tiny_pool_param %s_param ={\n" % op.name)
                    if  op.get_attr("data_format")=='NHWC':
                        w.write("           .kernel_h = %d,\n" % op.get_attr("ksize")[1])
                        w.write("           .kernel_w = %d,\n" % op.get_attr("ksize")[2])
                        w.write("           .stride_h = %d,\n" % op.get_attr("strides")[1])
                        w.write("           .stride_w = %d,\n" % op.get_attr("strides")[2])
                    elif op.get_attr("data_format")=='NCHW':
                        w.write("           .kernel_h = %d,\n" % op.get_attr("ksize")[2])
                        w.write("           .kernel_w = %d,\n" % op.get_attr("ksize")[3])
                        w.write("           .stride_h = %d,\n" % op.get_attr("strides")[2])
                        w.write("           .stride_w = %d,\n" % op.get_attr("strides")[3])
                    if op.get_attr("padding")=='SAME':
                        w.write("           .pad_h = NN_PAD_SAME,\n" )
                        w.write("           .pad_w = NN_PAD_SAME,\n" )
                    else: 
                        w.write("           .pad_h = NN_PAD_VALID,\n" )
                        w.write("           .pad_w = NN_PAD_VALID,\n" )
                    if op.type == 'MaxPool':
                        w.write("           .pool_method = NN_POOL_MAX,\n")
                    else : 
                        w.write("           .pool_method = NN_POOL_AVG,\n")
                    w.write("};\n\n")

                elif op.type == 'Conv2D':
#                    next_op = OP[OPS.index(op)+1]
                    w.write("static const struct tiny_conv_param %s_param ={\n" % op.name)
                    if  op.get_attr("data_format")=='NHWC':
                        w.write("           .kernel_h = %d,\n" %  op.inputs[1].shape.as_list()[1])
                        w.write("           .kernel_w = %d,\n" %  op.inputs[1].shape.as_list()[2])
                        w.write("           .stride_h = %d,\n" % op.get_attr("strides")[1])
                        w.write("           .stride_w = %d,\n" % op.get_attr("strides")[2])
                    elif op.get_attr("data_format")=='NCHW':
                        w.write("           .kernel_h = %d,\n" %  op.inputs[1].shape.as_list()[2])
                        w.write("           .kernel_w = %d,\n" %  op.inputs[1].shape.as_list()[3])
                        w.write("           .stride_h = %d,\n" % op.get_attr("strides")[2])
                        w.write("           .stride_w = %d,\n" % op.get_attr("strides")[3])
                    w.write("           .pad_h = NN_PAD_VALID,\n" )
                    w.write("           .pad_w = NN_PAD_VALID,\n" )
                    w.write("           .activation = -1,\n")
                    w.write("};\n\n")
                   
                    

############################# node structure#############################################
            flag = 0 
            for op in OPS:
                if op.type != 'Const' and op.type != 'Identity' :
                    if op.type == 'Add' or op.type == 'BiasAdd':  
                        if OPS[OPS.index(op)-1].type == 'Conv2D' or OPS[OPS.index(op)-1].type == 'DepthwiseConv2dNative' or OPS[OPS.index(op)-1].type == 'MatMul':  
                            flag = 1
                            next_input_name = op.inputs[0].name                   
                            if OPS[OPS.index(op)+1].type == 'FakeQuantWithMinMaxVars':  
                                next_input_name = OPS[OPS.index(op)+1].inputs[0].name                       
                            continue ;
                    if op.type == 'FakeQuantWithMinMaxVars':    #skip the fake quant operator
                            continue ;
                    w.write("static const struct tiny_node %s ={\n" % op.name)
                    #input number shoudl be update     
                    if op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative' or  op.type == 'MatMul' : 
                        next_op =  OPS[OPS.index(op)+1] 
                        if next_op.type == 'Add' or  next_op.type == 'BiasAdd':
                            w.write("           .input_num = %d,\n" % (len(op.inputs)+1))
                    else:         
                        w.write("           .input_num = %d,\n" % len(op.inputs))
                    ##############################
                    w.write("           .output_num = %d,\n" % len(op.outputs))
                    w.write("           .op_type = NN_OP_%s,\n" % str(op.type).upper())
                    w.write("           .op_ver = NN_OP_VERSION_1,\n")
                    if op.type == 'Conv2D' or op.type == 'MaxPool' or op.type == 'AvgPool':
                        w.write("           .op_param = &%s_param,\n" % op.name)
                    else :
                        w.write("           .op_param = NULL,\n")
                    ### input tensor
                    if  len(op.inputs) == 0:
                        w.write("           .input = NULL,\n")
                    else:
                        w.write("           .input = {" )
                        for index , t in enumerate (op.inputs):
                            if index == 0 and flag == 1:
                                w.write(" &%s,"% str(next_input_name).replace(':','_').replace('/','_') )
                                flag = 0
                            else: 
                                w.write(" &%s,"% str(t.name).replace(':','_').replace('/read','').replace('/','_') )
                        #merge Conv and Add
                        if op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative' or op.type == 'MatMul' : 
                            next_op =  OPS[OPS.index(op)+1] 
                            if next_op.type == 'Add' or  next_op.type == 'BiasAdd': 
                                w.write(" &%s,"% str(next_op.inputs[1].name).replace(':','_').replace('/read','').replace('/','_') )
                        w.seek(-1,1)
                        w.write("},\n")
                    
    
                    w.write("           .output = {" )
                    for ot in op.outputs:
                        w.write(" &%s,"%  str(ot.name).replace(':','_').replace('/','_') )
                    w.seek(-1,1) #move pointer backword
                    w.write("},\n")
                    w.write("};\n\n")

############################# Node list#############################################
            w.write("static const struct tiny_node* node_list[] = {\n")
            i = 0     
            for op in OPS:
                if op.type != 'Const' and op.type != 'Identity' :
                    if op.type == 'Add' or op.type == 'BiasAdd':  
                        if OPS[OPS.index(op)-1].type == 'Conv2D' or OPS[OPS.index(op)-1].type == 'DepthwiseConv2dNative' or OPS[OPS.index(op)-1].type == 'MatMul':  
                            continue ;
                    if op.type == 'FakeQuantWithMinMaxVars':
                            continue ;
                    w.write(" &%s," % op.name.ljust(12))     
                    i += 1
                    if i % 5 == 0 :    
                        w.write("\n")
            w.seek(-1,1) #move pointer backword
            w.write("};\n\n")
            
############################# graph structure#############################################
            w.write("static const struct tiny_graph tiny_graph ={\n")
            w.write("          .name = \"%s\",\n" % graph_name )
            w.write("          .tiny_version = NN_TINY_VERSION_1,\n")
            w.write("          .nn_id = 0xdeadbeaf,\n")
            w.write("          .create_time = %s,\n" % datetime.datetime.now().strftime("%Y%m%d"))
            w.write("          .layout = NN_LAYOUT_NHWC,\n")
            w.write("          .node_num = sizeof(node_list) / sizeof(void*),\n")
            w.write("          .node_list = node_list,\n")
            w.write("};\n\n")



            w.write("const struct tiny_graph* get_tiny_graph(void)\n")
            w.write("{\n")
            w.write("    return &tiny_graph;\n")
            w.write("}\n")
            w.write("\n")
            w.write("void free_tiny_graph(const struct tiny_graph* tiny_graph)\n")
            w.write("{\n")
            w.write("    /* NOTHING NEEDS TO DO */\n")
            w.write("}\n")

            w.close()
    return OPS


def read_graph_from_pb(tf_model_path):  
    with open(tf_model_path, 'rb') as f:
        serialized = f.read() 
    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(serialized) 
    print(graph_def)
            
    get_ops_from_pb_and_write_h(graph_def)
        
    print('\n')
    for node in graph_def.node:
        print(node.name)
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name='') 
    
    with tf.Session(graph=g) as sess: 
        OPS=get_ops_from_pb(g)
    return OPS        

if __name__=='__main__':
    model_path = sys.argv[1]
    print(model_path)
    if not sys.argv[2] :
        print ("No graph name input , use the default name \n")
    else :
        graph_name = sys.argv[2]
#    output_name = sys.argv[3]
#    html_dst = sys.argv[4]
#    input_names=input_names.split(',')
#    read_graph(model_path,input_names,output_name,html_dst)
#    read_graph(model_path)
    ops = read_graph_from_pb( model_path )
#    print(ops)	
#    gen_graph(ops)

