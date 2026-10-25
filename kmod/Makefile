# Out-of-tree build. For WSL2, KSRC points at the prepared Microsoft kernel
# tree (see scripts/wsl-setup.sh); on a normal distro the default works.
KSRC ?= /lib/modules/$(shell uname -r)/build

# A locally built vmlinux has different BTF from the running (Microsoft-built)
# kernel; module BTF generated against it is rejected noisily at load time.
NO_BTF := CONFIG_DEBUG_INFO_BTF_MODULES=

all:
	$(MAKE) -C $(KSRC) M=$(CURDIR) $(NO_BTF) modules

clean:
	$(MAKE) -C $(KSRC) M=$(CURDIR) clean
	rm -f ../core/*.o ../core/.*.cmd

.PHONY: all clean
