all:
	$(MAKE) -C v21
	$(MAKE) -C v46
	$(MAKE) -C v47
	$(MAKE) -C v70
	$(MAKE) -C v61
	$(MAKE) -C v60
	$(MAKE) -C vXXX

clean:
	$(MAKE) -C v21 clean
	$(MAKE) -C v46 clean
	$(MAKE) -C v47 clean
	$(MAKE) -C v70 clean
	$(MAKE) -C v61 clean
	$(MAKE) -C v60 clean
	$(MAKE) -C vXXX clean

.PHONY: all clean
