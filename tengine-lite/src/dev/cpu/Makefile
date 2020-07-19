obj-y+=op/

obj-y+=cpu_device.o
obj-y+=cpu_node_ops.o
obj-y+=cpu_module.o
obj-y+=cpu_probe.o

obj-$(CONFIG_HCL_BACKEND)+=hcl_module.o
obj-$(CONFIG_HCL_BACKEND)+=hcl_cpu.o

hcl_module_CFLAGS+=-I$(HCL_ROOT)/include
hcl_cpu_CFLAGS+=-I$(HCL_ROOT)/include
