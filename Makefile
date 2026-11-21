export PG_CONFIG ?= pg_config
PKGLIBDIR := $(shell $(PG_CONFIG) --pkglibdir)
SHAREDIR := $(shell $(PG_CONFIG) --sharedir)
INSTALL ?= install

.PHONY: all build clippy install uninstall

all: build

build:
	cargo run --locked -p xtask -- build

clippy:
	cargo run --locked -p xtask -- clippy

install:
	$(INSTALL) -d $(DESTDIR)$(PKGLIBDIR) $(DESTDIR)$(SHAREDIR)/extension && \
	$(INSTALL) -m 755 ./build/pkglibdir/* $(DESTDIR)$(PKGLIBDIR)/ && \
	$(INSTALL) -m 644 ./build/sharedir/extension/* $(DESTDIR)$(SHAREDIR)/extension/

uninstall:
	$(RM) $(wildcard $(DESTDIR)$(PKGLIBDIR)/vchord.*) && \
	$(RM) $(wildcard $(DESTDIR)$(SHAREDIR)/extension/vchord.*) && \
	$(RM) $(wildcard $(DESTDIR)$(SHAREDIR)/extension/vchord--*.sql)
