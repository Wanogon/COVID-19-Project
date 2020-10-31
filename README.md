# COVID-19-Project
This repository stores all codes needed in our paper <Group testing enables asymptomatic Screening for COVID-19 Mitigation 
-- Feasibility and Optimal Pool Size Selection with Dilution Effects>

dilution_effect_model.py solves our problem illustrates in section 3 so that we build the dilution effect model.

comparsion.py implements our comparsion on false negatives, false positives, tests per person between linear array test method and square array method in section 4.

gamma_eta_n_d.py and psi_theta_n_d.py estimates false negative rate for pool with size n and d positives, three correlation coefficients mentioned in section 4. The rounded results
are stored in the four .npy files: F_N_d_round.npy, Eta_N_d_round.npy, Psi_N_d_round.npy and Theta_N_d_round.npy, respectively.

Testing_cycle_simulation.py is our main file for implementing the testing-quarantine-infection model. In this file, closed_form_A2_approximate_10.py and pool_simulation.py are imported.
closed_form_A2_approximate_10.py is used to solve the optimization problem proposed in Eq.(11) in section 4, so that we decide the pool size of each testing cycle. It includes function
that solve the optimization problem using closed form solutions.
pool_simulation.py is used in the testing stage of each day, which contains functions that simulate the testing process of one linear array test and one square array test.
