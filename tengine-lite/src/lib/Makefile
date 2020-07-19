obj-y+=map.o
obj-y+=hash.o
obj-y+=hash_impl.o
obj-y+=vector.o
obj-y+=module.o
obj-y+=tengine_errno.o
obj-y+=tengine_log.o
obj-y+=tengine_c_api.o
obj-y+=tengine_ir.o
obj-y+=tengine_ir_pack.o
obj-y+=tengine_utils.o
obj-y+=tengine_exec.o
obj-y+=tengine_op.o
obj-y+=tengine_serializer.o
obj-$(CONFIG_TENGINE_PLUGIN)+=tengine_plugin.o
obj-y+=dev_allocator.o
obj-y+=exec_scheduler.o
obj-y+=nn_device.o
obj-y+=sys_port.o

#
obj-$(CONFIG_INTERN_ALLOCATOR)+=buddy_mem.o
obj-$(CONFIG_INTERN_ALLOCATOR_INIT)+=buddy_mem_init.o
obj-$(CONFIG_MEM_STAT)+=mem_stat.o

ifneq ($(CONFIG_INTERN_ALLOCATOR),)
    sys_port_CFLAGS+=-DCONFIG_INTERN_ALLOCATOR
    mem_stat_CFLAGS+=-DCONFIG_INTERN_ALLOCATOR
    buddy_mem_init_CFLAGS+=-DCONFIG_INTERN_ALLOCATOR
endif

ifneq ($(CONFIG_MEM_STAT),)
    sys_port_CFLAGS+=-DCONFIG_MEM_STAT
endif

ifneq ($(CONFIG_VERSION_POSTFIX),)
   tengine_c_api_CFLAGS+=-DCONFIG_VERSION_POSTFIX=$(CONFIG_VERSION_POSTFIX)
endif
