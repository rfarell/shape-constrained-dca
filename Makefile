PYTHON ?= python
THREADS = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
DEVICE ?= cpu
SEEDS = 42 43 44 45 46

.PHONY: test tables summary-figures verify prepare data-summary figures

# Need only the repository.
test:
	$(THREADS) $(PYTHON) -m pytest -q

tables:
	$(PYTHON) scripts/paper_results_bundle.py

summary-figures:
	$(PYTHON) scripts/paper_summary_figures.py \
		--arch-val-scores results/scores/arch/score_val_full/scores.json \
		--arch-test-scores results/scores/arch/score_test_full/scores.json \
		--arch-hindcast results/scores/arch/hindcast_test/hindcast.json \
		--arch-val-diversity results/scores/arch/val_diversity.json \
		--arch-test-diversity results/scores/arch/test_diversity.json \
		--arch-test-pairs results/scores/arch/score_test_pairs/diversity.json

# Need data/raw/bakken.csv (see data/README.md).
verify:
	$(PYTHON) scripts/audit_bakken.py --verify

prepare: verify
	$(PYTHON) scripts/prepare_bakken.py
	cmp data/split_manifest.json data/processed/bakken_v1/split_manifest.json
	@echo "Split matches data/split_manifest.json"

data-summary: prepare
	$(PYTHON) scripts/paper_data_summary.py

figures: prepare
	$(THREADS) $(PYTHON) scripts/paper_figures.py \
		$(foreach s,$(SEEDS),--bayes models/ensemble/bayes_arch_seed$(s)_model.pt) \
		$(foreach s,$(SEEDS),--basis models/comparators/basis_secondary_seed$(s)_model.pt) \
		$(foreach s,$(SEEDS),--dca models/comparators/dca_secondary_seed$(s)_model.pt) \
		--model-label arch --report-inflation 1.0 --prior-inflation 1.0 --device $(DEVICE)
