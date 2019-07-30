/base - !!! Lots of stuff in here, needs filtering and organising into training, generation and models !!!
/data
	/extracted_data - Beatsaber Levels and Songs
	/metadata - [PLACEHOLDER] Beastsaber complete metadata file (guillermo)
	/statespace - Contains outputs of identify_state_space.py (tim/ralph)
/models
/process_scripts
	/data_retrieval
		/io_functions.py - Check through
		/download_data.py - Perhaps integrate Beastsaber as download source
	/data_processing - lots of redundant code here, repetition in /base/models
		/difficulty_analysis.py [INCOMPLETE]
		/identify_state_space.py [DUPLICATE?] (level_processing_functions.py)
		/state_space_functions.py [DUPLICATE?] (level_processing_functions.py) 
	/feature_extraction - possible duplications in /base/models
		/example_decode_encode.py [LEGACY?] perhaps redundant (DEV testing code?)
		/feature_extraction.py
		/features_base.py - contains non-ML approach to level generation (should be extracted)
		/process_songs.py - needs to be integrated or deleted
	/training
	/generation
	/evaluation
		/rule_check_states.py - check/comment
		/graph_visualisation.py - check/comment 
	/misc
		/bash_scripts - Misc Bash scripts (Guillermo)
		/experiment_name - Unknown options file (Andrea/Guillermo)
/web - [DEV] The beginnings of a web api for level generation 
