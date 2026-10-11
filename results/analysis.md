# Task 6 — Analysis

## 6.1 Why do Set R logistic regression and the linear SVM score almost the same?

Both models use the same six standardised raw measurements and produce a linear decision boundary in that feature space. The Set R logistic regression validation accuracy is **73.08%**, while the linear SVM validation accuracy is **72.58%**. Their similar performance suggests that the main limitation is the linear boundary on these six raw features, rather than the specific choice between these two linear classifiers.

## 6.2 Signs of the Set Q squared-feature weights

All three current-squared weights are positive: `Ia^2 = 18.891`, `Ib^2 = 4.754`, and `Ic^2 = 13.957`. Holding the other features fixed, larger squared-current features push the model toward the fault class, which is consistent with fault conditions being associated with larger current magnitudes. All three voltage-squared weights are negative: `Va^2 = -6.028`, `Vb^2 = -6.464`, and `Vc^2 = -6.141`. Holding the other features fixed, larger squared-voltage features push the model away from the fault class, consistent with voltage sag during a fault. These are conditional model associations; correlated features mean that an individual coefficient should not be treated as a standalone physical law.

## 6.3 Support vectors in the best linear and RBF SVMs

The selected linear SVM (`C = 0.1`) has **5,849 support vectors**, while the selected RBF SVM (`C = 100`, `gamma = 1`) has **266**. A very large support-vector count means many training examples are involved in defining the margin/decision function. In this dataset, the linear model's large count and modest validation accuracy are consistent with a linear boundary struggling to represent the class structure.

## 6.4 What happens as C increases?

For the linear SVM, validation accuracy stays at **72.58%** for all four tested C values; support-vector counts change only slightly (5,849, 5,836, 5,835, and 5,836). For the RBF SVM, accuracy improves across the tested settings, reaching **99.67%** at `C = 100`, `gamma = 1`. The support-vector count generally drops as C increases for a fixed gamma in these results, although this is not a universal guarantee. C controls the penalty for training errors: a larger C places more emphasis on classifying training examples correctly and can produce a more tightly fitted boundary.

## 6.5 Is the random-split test accuracy optimistic or pessimistic for a real relay?

It is likely optimistic. Rows are consecutive samples, and the required random split can place highly similar neighbouring samples in both training and test sets. The test set may therefore be easier than genuinely future measurements from a new fault event or different operating conditions. A time-ordered or event-level evaluation would be a stronger check for a real protection system, but the assignment's required random split is retained here.

## 6.6 Which model would we deploy?

Among these four tested models, the RBF SVM has the strongest test results: **99.58% accuracy, 6 missed faults, and 4 false alarms**. The Set Q logistic regression has 45 missed faults and 6 false alarms; the linear models miss substantially more faults. Based on the assignment's emphasis on missed faults and false alarms, the RBF SVM is the strongest candidate among the evaluated models. However, this dataset and random split do not certify a real relay: further event-level/time-based testing, engineering validation, and safety certification would be required before deployment.
