# SDM

I created this structure to help me with a solve a project. I chose to call it a set decision matrix because, for its intended purpose, there are no duplicate values ("set"), and maintains a reference to the current branch of the tree, or the best *decision* to take at that point in time (which is, although a bit of a stretch, the general purpose of a "decision matrix"). Whenever a duplicate value is inputted, that node becomes the current branch.

I built upon the more generalized SDM and made the TargetSDM, which is initialized with a target value. Depending on the mode, the structure either only allows inputs that are closer to the target than the current best, or inputs any value but maintains a reference to the current best. A boolean flag is triggered when the value is reached.

The TargetSDM has to be initialized with a KeyFormula. I made some default ones, but you may want to create a custom one. This is akin to overloading the '>' and '<' operators as you would with custom objects in a generic tree structure. This is because it has to be clear how your object should be "graded", or computed into a more digestible key which the structure compares against when determining improvement (or the lack thereof).
