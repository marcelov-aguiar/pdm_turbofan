import random
import numpy as np
import threading
from sklearn.model_selection import train_test_split


class Population:
    def __init__(
            self,
            individual_constructor,
            pop_size,
            n_parents,
            use_recombination=True,
            mutation_rate=0.01,
            deterministic_selection_percentage=0.7,
            multithread=True):
        self.__individual_factory = IndividualFactory(individual_constructor)
        self.__pop_size = pop_size
        self.__n_parents = n_parents
        self.__use_recombination = use_recombination
        self.__mutation_rate = mutation_rate
        self.__use_multithreading = multithread
        self.__n_parents_to_select_deterministically = round(
            deterministic_selection_percentage * self.__n_parents)
        self.__n_parents_to_select_randomly = self.__n_parents - \
            self.__n_parents_to_select_deterministically
        self.__individuals = self.__create_individuals()
        self.__parents = []
        self.__kids = []
        self.__X = None
        self.__y = None
        self.__test_size = None

    def set_data(self, X, y, test_size=0.2):
        self.__X = X
        self.__y = y
        self.__test_size = test_size

    def get_individuals(self):
        return self.__individuals

    def get_best_individual(self):
        return max(
            self.__individuals,
            key=lambda individual: individual.get_fitness())

    def get_oldest_individual(self):
        return max(
            self.__individuals,
            key=lambda individual: individual.get_age())

    def get_fitness_percentiles(
        self,
        percentiles=[
            0,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100]):
        fitness_array = [individual.get_fitness()
                         for individual in self.__individuals]
        fitness_percentiles = np.percentile(fitness_array, percentiles)
        percentiles_dict = {}
        for i in range(len(percentiles)):
            percentiles_dict[str(percentiles[i])] = fitness_percentiles[i]
        return percentiles_dict

    def fit(self):
        if self.__use_multithreading:
            self.__fit_with_multithread()
        else:
            self.__fit_without_multithread

    def select_parents(self):
        self.__sort_by_fitness()
        self.__parents = \
            self.__individuals[-self.__n_parents_to_select_deterministically:]
        remaining_individuals = \
            self.__individuals[:-self.__n_parents_to_select_deterministically]
        self.__parents.extend(
            random.sample(
                remaining_individuals,
                k=self.__n_parents_to_select_randomly))
        self.__increase_parents_age()

    def reproduce(self):
        self.__kids = []
        if self.__use_recombination:
            self.__reproduce_with_recombination()
        else:
            self.__reproduce_without_recombination()

    def mutate(self):
        for individual in self.__kids:
            if random.random() < self.__mutation_rate or \
               not self.__use_recombination:
                individual._mutate()

    def join_generation(self):
        self.__individuals = []
        self.__individuals = self.__parents + self.__kids

    def __create_individuals(self):
        return self.__individual_factory.make_individuals(self.__pop_size)

    def __increase_parents_age(self):
        for parent in self.__parents:
            parent._increase_age()

    def __fit_with_multithread(self):
        X_train, X_val, y_train, y_val = \
            self.__split_train_and_validation_data()

        threads = list(map(lambda individual: FitThread(
            individual, X_train, X_val, y_train, y_val), self.__individuals))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def __fit_without_multithread(self):
        X_train, X_val, y_train, y_val = \
            self.__split_train_and_validation_data()

        for individual in self.__individuals:
            individual._fit(X_train, y_train, X_val, y_val)

    def __split_train_and_validation_data(self):
        return train_test_split(self.__X, self.__y, test_size=self.__test_size)

    def __sort_by_fitness(self):
        self.__individuals.sort(
            key=lambda individual: individual.get_fitness())

    def __reproduce_with_recombination(self):
        while len(self.__parents) + len(self.__kids) < self.__pop_size:
            mom = random.choice(self.__parents)
            dad = random.choice(self.__parents)
            self.__breed_individual_with_recombination(mom, dad)

    def __breed_individual_with_recombination(self, mom, dad):
        kid = self.__individual_factory.make_individual()
        kid._gene = [random.choice(pair) for pair in zip(mom._gene, dad._gene)]
        self.__kids.append(kid)

    def __reproduce_without_recombination(self):
        n_individuals_to_choose = self.__pop_size - self.__n_parents
        individuals_to_clone = random.sample(
            self.__parents, n_individuals_to_choose)
        for individual in individuals_to_clone:
            self.__kids.append(individual._clone())

        while len(self.__parents) + len(self.__kids) < self.__pop_size:
            self.__kids.append(random.choice(self.__parents)._clone())


class IndividualFactory:
    def __init__(self, individual_constructor):
        self.__individual_constructor = individual_constructor

    def make_individuals(self, n_individuals):
        return [self.make_individual() for n in range(n_individuals)]

    def make_individual(self):
        return self.__individual_constructor()


class FitThread (threading.Thread):
    def __init__(self, individual, X_train, X_val, y_train, y_val):
        threading.Thread.__init__(self)
        self.individual = individual
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def run(self):
        self.individual._fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val)


class FeatureSelectionIndividual:
    def __init__(
            self,
            model_constructor,
            possible_features,
            validation_score_weigth=0.5,
            scoring_function=None,
            past_scores_to_keep=1,
            extra_feature_penalty=0,
            features_without_penalty=0):
        self.__age = 0
        self.__is_fitted = False
        self.__possible_features = possible_features
        self.__scoring_function = scoring_function
        self.__past_scores_to_keep = past_scores_to_keep
        self.__validation_score_weigth = validation_score_weigth
        self.__extra_feature_penaly = extra_feature_penalty
        self.__features_without_penalty = features_without_penalty
        self.__model_constructor = model_constructor
        self.__model = model_constructor()

        self._gene = [False for feature in possible_features]
        self._past_scores = []

    def get_used_features(self):
        used_features = [self.__possible_features[i]
                         for i in range(len(self._gene)) if self._gene[i]]
        return used_features

    def get_model(self):
        return self.__model

    def get_fitness(self) -> float:
        return np.mean(self._past_scores) if len(self._past_scores) > 0 else 0

    def get_age(self):
        return self.__age

    def _clone(self):
        clone = FeatureSelectionIndividual(
            self.__model_constructor,
            self.__possible_features,
            self.__validation_score_weigth,
            self.__scoring_function,
            self.__past_scores_to_keep)
        clone._gene = self._gene
        return clone

    def _increase_age(self):
        self.__age = self.__age + 1

    def _fit(self, X_train, y_train, X_val, y_val):
        self.__mutate_if_doesnt_use_any_features()
        X_train = self.__filter_dataframe_for_selected_columns(X_train)
        X_val = self.__filter_dataframe_for_selected_columns(X_val)

        if not self.__is_fitted:
            self.__model.fit(X_train.values, y_train.values.ravel())
            self.__is_fitted = True

        self.__calculate_score(X_train, y_train, X_val, y_val)

    def _mutate(self):
        random_index = random.randint(0, len(self._gene) - 1)
        self._gene[random_index] = not self._gene[random_index]

    def __filter_dataframe_for_selected_columns(self, data):
        return data.iloc[:, self._gene]

    def __mutate_if_doesnt_use_any_features(self):
        if not any(self._gene) and not self.__is_fitted:
            self._mutate()

    def __calculate_score(self, X_train, y_train, X_val, y_val):
        score = None
        if self.__scoring_function:
            score = (1 - self.__validation_score_weigth) * \
                    self.__scoring_function(y_train,
                                            self.__model.predict(X_train)) + \
                    (self.__validation_score_weigth) * \
                    self.__scoring_function(y_val, self.__model.predict(X_val))
        else:
            score = (1 - self.__validation_score_weigth) * \
                    self.__model.score(X_train, y_train) + \
                    (self.__validation_score_weigth) * \
                    self.__model.score(X_val, y_val)

        extra_features = max(0, sum(self._gene) -
                             self.__features_without_penalty)
        penalty = self.__extra_feature_penaly * extra_features
        score = score - penalty

        self._past_scores.append(score)
        if len(self._past_scores) > self.__past_scores_to_keep:
            self._past_scores = self._past_scores[-self.__past_scores_to_keep:]
