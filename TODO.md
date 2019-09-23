

# TODOs

- [ ] GUI/website
- [ ] Save the stage 1 probabilities and have interactive visualizer as you change threshold.
- [ ] Finish writing main README
- [ ] Finish writing training README, for stage_two.
- [ ] stage_two dataset uses `state_space_functions`, while general_beat_saber dataset uses `level_processing_functions`. Check if need to merge these two, as they have a fair of common functionality I think.
- [ ] `level_processing_functions` is even more ugly now, because of new fix to speed up training by preprocessing levels into numpy tensors, and leaving the old functions just in case (feature creep :P). So should remove the old functions, as memory is cheaper than computing generally, so preprocessing is generally preferred if it speeds up computing!
- [x] is there a way go make `get_reduced_tensors_from_level` in `level_processing_functions` faster? -- now that we've fixed the bug, and made the sequence length longer, it's slowed down training :(. Yes there is, by preprocessing, but it introduced new TODO above!
- [x] make stage_two_dataset work with new beat saber data.
- [x] fix importing. E.g. base_options.py imports from `base`, that doesn't exist now....
- [x] Make DDC port
- [x] Train on new data
- [ ] Testing code. Use perplexity, as in NLP literature
- [ ] Make stage two that uses multi_mel features, or even the DDC embeddings as its inputs
- [ ] Make non-reduced-state data work, wavenet.
- [ ] Obstacles, etc.

---

## Folder structure and refactoring comments

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
