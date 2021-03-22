| Experiement | Accuracy | Confusion Matrix                  | Comment |
|-------------|----------|-----------------------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] | |
| Solution 1   | 0.7657992565055762  | [[159  15] [ 48  47]] |  In this first iteration i have taken glucose to predict and test size as 0.35 |
| Solution 2   | 0.7835497835497836  | [[132  14] [ 36  49]] |  In this second iteration i have taken glucose,bmi and age as my 3 parameters and taken test size as 0.3 |
| Solution 3   | 0.8008658008658008  | [[133  13] [ 33  52]] | I have taken bp,glucose,bmi and replaced the 0 value with mean of the values and took test size as 0.3 |
