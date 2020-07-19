obj-$(CONFIG_OP_CONV)+=convolution.o
obj-$(CONFIG_OP_POOL)+=pooling.o
obj-$(CONFIG_OP_FC)+=fc.o
obj-y+=simple_op.o

ifeq ($(CONFIG_DISABLE_PARAM_ACCESS),y)
    COMMON_CFLAGS+=-DCONFIG_DISABLE_PARAM_ACCESS
endif

