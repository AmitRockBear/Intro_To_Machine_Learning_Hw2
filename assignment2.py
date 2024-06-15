#################################
# Your name: Amit Rockach
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        samples = np.sort(np.random.uniform(0, 1, m))
        labels = [np.random.choice([0, 1], p=self.get_y_probabilities_by_x(x)) for x in samples]
        return np.column_stack((samples, labels))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        ms = list(range(m_first, m_last + 1, step))
        empirical_errors, true_errors = self.experiment_m_range_erm_calculations(ms, k, T)
        plt.plot(ms, empirical_errors, label='Empirical Error')
        plt.plot(ms, true_errors, label='True Error')
        plt.xlabel('m')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

        return np.column_stack((empirical_errors, true_errors))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        ks = list(range(k_first, k_last + 1, step))
        empirical_errors, true_errors = self.experiment_k_range_erm_calculations(m, ks)
        plt.plot(ks, empirical_errors, label='Empirical Error')
        plt.plot(ks, true_errors, label='True Error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

        return ks[np.argmin(empirical_errors)]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        ks = list(range(k_first, k_last + 1, step))
        empirical_errors, true_errors = self.experiment_k_range_erm_calculations(m, ks)

        penalties = [math.sqrt((2 * k +  math.log((2 * k**2) / 0.1)) / m) for k in ks]
        penalty_emperical_error_sum_list = [penalties[i] + empirical_errors[i] for i in range(len(ks))]

        plt.plot(ks, empirical_errors, label='Empirical Error')
        plt.plot(ks, true_errors, label='True Error')
        plt.plot(ks, penalties, label='Penalty')
        plt.plot(ks, penalty_emperical_error_sum_list, label='Penalty + Empirical Error')
        plt.xlabel('k')
        plt.legend()
        plt.show()

        return ks[np.argmin(penalty_emperical_error_sum_list)]

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples = self.sample_from_D(m)
        ks = list(range(1, 11))
        xs, ys = samples[:, 0], samples[:, 1]
        train_size = int(0.8 * m)
        xs_train, ys_train = xs[:train_size], ys[:train_size]
        xs_validate, ys_validate = xs[train_size:], ys[train_size:]
        validation_errors = []
        
        for k in ks:
            hypothesis, _ = intervals.find_best_interval(xs_train, ys_train, k)
            validation_errors.append(self.calculate_validation_error(xs_validate, ys_validate, hypothesis))
        
        return ks[np.argmin(validation_errors)]

    #################################
    # Place for additional methods
    def get_y_probabilities_by_x(self, x):
        if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
            return [0.2, 0.8]
        return [0.9, 0.1]

    def calculate_hypothesis_true_error(self, I):
        def interval_intersection(interval1, interval2):
            start1, end1 = interval1
            start2, end2 = interval2
            start_intersection = max(start1, start2)
            end_intersection = min(end1, end2)
            if start_intersection <= end_intersection:
                return (start_intersection, end_intersection)
            else:
                return None

        def find_intersections_and_leftovers(list1, list2):
            intersections = []
            leftovers = []
            for interval2 in list2:
                interval_intersections = []
                for interval1 in list1:
                    intersection = interval_intersection(interval1, interval2)
                    if intersection:
                        interval_intersections.append(intersection)
                if len(interval_intersections) == 0:
                    leftovers.append(interval2)  # Add interval2 to leftovers if it does not intersect with any interval in list1
                else:
                    intersections.extend(interval_intersections)
                    # Calculate leftovers of interval2
                    leftover_intervals = []
                    start2, end2 = interval2
                    for start1, end1 in sorted(interval_intersections):
                        if start2 < start1:
                            leftover_intervals.append((start2, start1))
                        start2 = end1
                    if start2 < end2:
                        leftover_intervals.append((start2, end2))
                    leftovers.extend(leftover_intervals)

            return intersections, leftovers

        y_1_intervals = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        unique_intervals = sorted(list(set(I)), key=lambda t: t[0])
        sum = 0

        # Calculate the true error
        # Calculate the intersections and leftovers of (unique_intervals-y_1_intervals)
        intersections, leftovers = find_intersections_and_leftovers(y_1_intervals, unique_intervals)
        for interval in intersections:
            sum += (interval[1] - interval[0]) * 0.2
        for interval in leftovers:
            sum += (interval[1] - interval[0]) * 0.9

        # Calculate leftovers of (y_1_intervals-unique_intervals)
        _, leftovers = find_intersections_and_leftovers(unique_intervals, y_1_intervals)
        for interval in leftovers:
            sum += (interval[1] - interval[0]) * 0.8
        
        # Calculate leftovers of ([(0,1)]-y_1_intervals-unique_intervals)
        _, leftovers = find_intersections_and_leftovers(y_1_intervals, [(0, 1)])
        _, leftovers = find_intersections_and_leftovers(unique_intervals, leftovers)
        for interval in leftovers:
            sum += (interval[1] - interval[0]) * 0.1
        
        return sum
    
    def experiment_m_range_erm_calculations(self, ms, k, T):
        empirical_errors = []
        true_errors = []

        for m in ms:
            empirical_errors_sum = 0
            true_errors_sum = 0
            for t in range(T):
                data = self.sample_from_D(m)
                xs, ys = data[:, 0], data[:, 1]
                hypothesis, empirical_error_count = intervals.find_best_interval(xs, ys, k)
                empirical_errors_sum+=empirical_error_count / m
                true_errors_sum+=self.calculate_hypothesis_true_error(hypothesis)
            empirical_errors.append(empirical_errors_sum / T)
            true_errors.append(true_errors_sum / T)
        
        return empirical_errors, true_errors

    def experiment_k_range_erm_calculations(self, m, ks):
        data = self.sample_from_D(m)
        empirical_errors = []
        true_errors = []

        for k in ks:
            xs, ys = data[:, 0], data[:, 1]
            hypothesis, empirical_error_count = intervals.find_best_interval(xs, ys, k)
            empirical_errors.append(empirical_error_count / m)
            true_errors.append(self.calculate_hypothesis_true_error(hypothesis))
        
        return empirical_errors, true_errors
    
    def calculate_validation_error(self, xs_validate, ys_validate, hypothesis):
        validation_error_count = 0
        validate_size = len(xs_validate)

        for i in range(validate_size):
            x = xs_validate[i]
            y = ys_validate[i]
            found = False
            for interval in hypothesis:
                if interval[0] <= x <= interval[1] and y == 0:
                    validation_error_count+=1
                    found = True
            if not found and y == 1:
                validation_error_count+=1

        return validation_error_count / validate_size
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

