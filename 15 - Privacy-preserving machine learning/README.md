# Privacy Preserving Machine Learning
This is the companion code package for the whitepaper "Privacy Preserving Machine Learning" published at the [actuarialdatascience.org](https://actuarialdatascience.org/ADS-Tutorials/) site.

Authors: Daniel Meier (Swiss Re Ltd), Michael Mayer (La Mobilière), members of the Data Science Working Group of the Swiss Association of Actuaries, see https://actuarialdatascience.org, and Juan-Ramón Troncoso-Pastoriza (Tune Insight).

The included code is split in:

- A notebook that introduces privacy preserving machine learning methods using synthetic health datasets with risk factors like BMI, blood pressure, age, etc. to predict various health outcomes of individuals over time. Besides the creation of synthetic datasets and the use and explainability of traditional and neural network models for mortality calculations, the notebook also introduces encrypted processing of the aforementioned models to enable processing real data that cannot be accessed.
- A Dockerfile for running the previous notebook within a Docker container.
- A (static, HTML) version of an advanced notebook that showcases the use of encrypted federated processing to enable compliant training of mortality models on data that is split across multiple sources and that cannot be shared or exported.

The encrypted computations are supported by Tune Insight's secure analytics and machine learning platform, and the first notebook makes use of Tune Insight's client side library, also included with this package.

## Running the example notebook and installing the hefloat package

Open the notebook `PPML.ipynb` in Jupyter or your favorite notebook viewer (e.g. vs-code).
The last section of the notebook exemplifies homomorphic encrypted processing with Tune Insight's client-side library.
The first cell of that section will install the hefloat package and its dependencies.

## Alternatively, using docker

You can run the Part 5 Homomorphic encryption of the notebook with Jupyter using docker

In the terminal, run the following command:
```bash
docker run -p 10000:8888 registry.tuneinsight.com/tuneinsight-exp/ti-hefloat-ppml
```

Open the following link in your browser:
http://localhost:10000/lab/tree/PPML-HE.ipynb

## Contact

If you have any questions or you want to know more about the full secure analytics and machine learning platform, contact us at [contact@tuneinsight.com](mailto:contact@tuneinsight.com).