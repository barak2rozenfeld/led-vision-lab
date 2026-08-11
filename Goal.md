# LED Vision Lab — Goal

## Vision

Build a Computer Vision platform that detects whether an LED is ON or OFF from an image and compares multiple detection approaches on the same problem.

The project should combine:

- Computer Vision
- Machine Learning
- Deep Learning
- Synthetic Data Generation
- Benchmarking
- REST API
- Web Application

The system should be useful both as an academic final project and as a foundation for a potential commercial product.

---

## Main Problem

In hardware testing, integration, validation, manufacturing, and automated test environments, engineers often need to verify visual indicators such as LEDs.

A typical flow is:

Test System  
→ Send command to DUT  
→ LED should turn ON  
→ Camera captures image  
→ Vision system detects LED state  
→ Expected vs. Actual  
→ PASS / FAIL

The project will explore which detection method provides the best trade-off between:

- Accuracy
- Speed
- Simplicity
- Explainability
- Robustness
- Training effort
- Deployment cost

---

## Core Research Question

**What is the simplest, fastest, and most reliable method for detecting whether an LED is ON or OFF under different visual conditions?**

---

## Detection Approaches

The project will compare:

1. HSV / Color Detection
2. Template / Image Difference
3. K-Nearest Neighbors
4. Small CNN

All methods should be evaluated using the same benchmark data and metrics.

---

## Synthetic Dataset

The project will include a synthetic LED dataset generation pipeline.

The generator should support scalable generation up to approximately:

**100,000 labeled images**

Initial labels:

- LED ON
- LED OFF

Synthetic variations may include:

- LED brightness
- LED color
- LED size
- LED position
- Camera angle
- Camera distance
- Ambient light
- Exposure
- Blur
- Noise
- Background
- Reflections
- Contrast

The dataset should be reproducible using configuration files and random seeds.

---

## Platform

The final platform should allow a user to:

- Upload an image
- Select a detection method
- Detect LED ON/OFF
- Compare multiple detectors
- View runtime and confidence
- Understand how each detector works
- View benchmark results
- Access the same functionality through an API

---

## Educational Goal

The platform should explain the differences between detection approaches.

For each method, the project should document:

- How it works
- Advantages
- Limitations
- Training requirements
- Runtime characteristics
- Recommended use cases

The goal is not only to provide a result, but also to demonstrate why one approach may be better than another.

---

## Engineering Goal

The project should demonstrate production-oriented engineering practices including:

- Clean architecture
- Modular detector interface
- Automated tests
- Reproducible experiments
- Version control
- CI
- API design
- Deployment
- Documentation

---

## Commercial Direction

The project should be designed so that the detection engine may later be offered as a service.

Possible future directions:

- Public web demo
- Developer API
- Usage limits
- API keys
- Paid API tiers
- Batch processing
- Local / on-premise deployment
- Custom models for hardware companies

Commercial functionality is secondary to proving the technical value of the system.

---

## Success Criteria

The project is considered successful when it can demonstrate:

- Reproducible synthetic data generation
- Multiple LED detection techniques
- Fair benchmarking across all methods
- Working REST API
- Working web interface
- Clear technical documentation
- Measurable comparison between methods
- A path toward real-world hardware validation

---

## Guiding Principle

Do not use Deep Learning simply because it is more advanced.

The project should identify the best engineering solution for each environment.

The key question is:

**When is a simple Computer Vision solution enough, and when is Machine Learning or Deep Learning justified?**