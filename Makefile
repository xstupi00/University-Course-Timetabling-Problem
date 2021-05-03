# Author: Simon Stupinsky

SRC_DIR := src
IN_DIR := inputs
OUT_DIR := outputs

DOC := doc/doc.pdf
PACK := xstupi00.tar.gz

.PHONY: run
run:
	python3 $(SRC_DIR)/ucttp.py --instance $(INSTANCE)

.PHONY: run_experiments
experiments:
	python3 $(SRC_DIR)/experiments.py

.PHONY: generate_graphs
graphs:
	python3 $(SRC_DIR)/graph.py


.PHONY: pack
pack: $(PACK)

$(PACK): Makefile README.md requirements.txt $(DOC) $(SRC_DIR) $(IN_DIR) \
		$(OUT_DIR)
	make clean
	COPYFILE_DISABLE=1 tar -czf $@ $^

.PHONY: clean
clean:
	rm -rf $(SRC_DIR)/*.pyc $(SRC_DIR)/__pycache__/ $(PACK)