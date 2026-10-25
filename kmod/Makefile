# Out-of-tree build. For WSL2, KSRC points at the prepared Microsoft kernel
# tree (see scripts/wsl-setup.sh); on a normal distro the default works.
KSRC ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KSRC) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KSRC) M=$(CURDIR) clean
	rm -f ../core/*.o ../core/.*.cmd

.PHONY: all clean
