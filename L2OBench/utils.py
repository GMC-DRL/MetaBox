from problem import bbob, fsr, protein_docking


def construct_problem_set(config):
    problem = config.problem
    if problem in ['bbob', 'bbob-noisy']:
        return bbob.BBOB_Dataset.get_datasets(suit=config.problem,
                                              dim=config.dim,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty)
    elif problem == 'FSR':
        return fsr.FSR_Dataset.get_datasets(train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=config.difficulty)
    elif problem == 'protein':
        return protein_docking.Protein_Docking_Dataset.get_datasets(train_batch_size=config.train_batch_size,
                                                                    test_batch_size=config.test_batch_size,
                                                                    difficulty=config.difficulty)
    else:
        raise ValueError(problem + ' is not defined!')
