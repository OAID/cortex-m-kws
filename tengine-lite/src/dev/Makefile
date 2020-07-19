obj-y+=cpu/

COMMON_CFLAGS+=-I$(shell pwd)/include

ifeq ($(CONFIG_HCL_BACKEND),y)
    COMMON_CFLAGS+=-DCONFIG_HCL_BACKEND
endif
