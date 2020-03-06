# Configuración

Este directorio contiene los ficheros de configuración del modelo.

## Configuración General

El fichero **config.cfg** contiene los parámetros del modelo:

    [Model]
     PAYOFF_MATRIX :  [[1 0] [0 2]] ,
     N_OF_AGENTS :  200 ,
     RANDOM_INITIAL_CONDITION : False,
     INITIAL_CONDITION :  [200 0] ,
     CANDIDATE_SELECTION :  direct ,
     N_OF_CANDIDATES :  2 ,
     DECISION_METHOD :  best ,
     COMPLETE_MATCHING : False,
     N_OF_TRIALS :  1 ,
     SINGLE_SAMPLE : False,
     TIE_BREAKER :  uniform ,
     LOG_NOISE_LEVEL :  0 ,
     USE_PROB_REVISION : True,
     PROB_REVISION :  0.1 ,
     N_OF_REVISIONS_PER_TICK :  1 ,
     PROB_MUTATION :  0 ,
     TRIALS_WITH_REPLACEMENT : False,
     SELF_MATCHING : True,
     IMITATEES_WITH_REPLACEMENT : False,
     CONSIDER_IMITATING_SELF : False,
     PLOT_EVERY_SECS :  0.526 ,
     DURATION_OF_RECENT :  10 ,
     SHOW_RECENT_HISTORY : True,
     SHOW_COMPLETE_HISTORY : True