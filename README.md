# MTA-Ridership-Predictor

  This project aims to expand upon the current literature and 
facilitate a proactive approach in fortifying the robustness of a complex transportation
system, primarily through exploring the various features of its ridership. Specifically,
the project seeks to identify the most pertinent features of the MTA system and employ
predictive models in order to best predict ridership rates. In doing so, the model could
provide a nuanced understanding of the subway system’s functioning and offer valuable
insights for urban planning and meeting passenger demand.
  The dataset used in this paper is directly from the Metropolitan Transportation
Authority (MTA) through the NYC Open Data platform. The dataset is updated daily
and provides subway ridership estimates on an hourly basis from February 2022 through
November 2023
  Considering the apparent non-linear nature of the data, the selected method ultimately involved Random Forest
Regression. Random Forest is a widely-used machine learning algorithm that combines
the output of multiple decision trees to reach a single result. Capable of
addressing both classification and regression problems, Random Forest is able to obtain
better predictive performance than traditional regression, while also inherently
protecting against overfitting as well as detecting nonlinear effects and interactions
among predictors. This also means that Random Forest
exhibits less susceptibility to multicollinearity.
  Ultimately, the Random Forest model achieved a R^2 value of 0.87, which suggests that the model accounted for 87% of the variability in MTA
subway ridership. Such a R2 is generally considered strong and indicates that the model
is an appropriate fit for the data.
