env:
	conda create -n llm-networks-misinformation python=3.10 -y
	@echo "Run conda activate llm-networks-misinformation"
deps:
	pip install -r requirements.txt
lab:
	jupyter lab