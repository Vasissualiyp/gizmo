# Standalone check of cooling/jaco_eos_anchor.h, the EOS a gas cell carries between jaco solves, against jaco's full
# u -> T inversion: regenerate the jaco EOS for MODEL into build/MODEL/, compile the test against it, run it. Exits
# non-zero on any failure.
#
#   make -C test/jaco_eos_anchor [MODEL=starforge_legacy]   # needs the jaco package importable by $(PYTHON)
#   make -C test/jaco_eos_anchor MAX_DU=0.3                  # the anchor's bound overridden, as Config.sh can

PYTHON ?= python3
MODEL  ?= starforge
MAX_DU ?=
BUILD  := build/$(MODEL)
CXXFLAGS ?= -O2 -g -Wall
CPPFLAGS += -I$(BUILD) -I../../cooling $(if $(MAX_DU),-DJACO_EOS_PREDICT_MAX_DU=$(MAX_DU))

.PHONY: test codegen clean FORCE
test: $(BUILD)/test_jaco_eos_anchor
	./$<

# Always rerun: the codegen only rewrites files whose content changed, so a stale build/ cannot hide a model change.
codegen:
	$(PYTHON) -m jaco.codegen.gizmo.gizmo $(MODEL) --language c --ext .cc --output-dir $(BUILD)
$(BUILD)/jaco_eos.cc $(BUILD)/microphysics_func_jac.h: codegen ;

# rewritten only when the flags change, so that changing MAX_DU rebuilds
$(BUILD)/flags: FORCE | codegen
	@echo '$(CXX) $(CPPFLAGS) $(CXXFLAGS)' | cmp -s - $@ || echo '$(CXX) $(CPPFLAGS) $(CXXFLAGS)' > $@

$(BUILD)/jaco_eos.o: $(BUILD)/jaco_eos.cc $(BUILD)/flags
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fno-associative-math -c $< -o $@
$(BUILD)/test_jaco_eos_anchor: test_jaco_eos_anchor.cc ../../cooling/jaco_eos_anchor.h $(BUILD)/jaco_eos.o $(BUILD)/flags
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) test_jaco_eos_anchor.cc $(BUILD)/jaco_eos.o -o $@

clean:
	rm -rf build
