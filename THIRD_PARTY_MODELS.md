# Modeles tiers optionnels

Les poids ne sont pas inclus dans ce depot et ne sont jamais telecharges pendant une
analyse par defaut.

## Community Forensics

- code et methode : <https://github.com/JeongsooP/Community-Forensics>
- poids officiels 384 : <https://huggingface.co/OwensLab/commfor-model-384>
- poids officiels 224 : <https://huggingface.co/OwensLab/commfor-model-224>
- publication : *Community Forensics: Using Thousands of Generators to Train Fake
  Image Detectors*, CVPR 2025
- licence declaree par le code et les fiches de modeles amont : MIT

Frod epingle la revision Hugging Face de chaque variante. Pour un checkpoint local,
le rapport identifie la version par un prefixe de son SHA-256. La sortie sigmoid est
nommee `synthetic_score`; elle ne doit pas etre interpretee comme une probabilite de
fraude.
