/base - !!! Lots of stuff in here, needs filtering and organising into training, generation and models !!!
/data
	/extracted_data - Beatsaber Levels and Songs
	/metadata - [PLACEHOLDER] Beastsaber complete metadata file (guillermo)
	/statespace - Contains outputs of identify_state_space.py (tim/ralph)
/models
/scripts
	/data_retrieval
		/download_data.py - Perhaps integrate Beastsaber as download source
	/data_processing - lots of redundant code here, repetition in /base/models
		/difficulty_analysis.py [INCOMPLETE]
		/state_space_functions.py (level_processing_functions.py)
			produce_distinct_state_space_representations()
			compute_explicit_states_from_json()
			compute_explicit_states_from_bs_level() - Wrapper
			compute_shortest_inter_event_beat_gap()
			produce_transition_probability_matrix_from_distinct_state_spaces() 
			compute_state_sequence_representation_from_json() - Sorts output of produce_distinct_state_space_representations for use in get_block_sequence_with_deltas()
			get_block_sequence_with_deltas()
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
	    /io_functions.py - Check through
		/bash_scripts - Misc Bash scripts (Guillermo)
		/experiment_name - Unknown options file (Andrea/Guillermo)
/web - [DEV] The beginnings of a web api for level generation

---
fix importing. E.g. base_options.py imports from `base`, that doesn't exist now....

make stage_two_dataset work with new beat saber data.

stage_two uses state_space_functions, while general_beat_saber uses level_processing_functions. Need to merge these two.
Finish writing training readme, for stage_two.
Make non-reduced-state data work
write block placement testing stuff
