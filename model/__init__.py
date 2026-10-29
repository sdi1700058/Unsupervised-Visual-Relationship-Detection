"""FOSAE: the network (model.py), the loaders, the training and the latent codes.

This file imports nothing. So numpy-only code, such as the planner, can read
an export through model.latents.export without Keras. model.loaders still
loads Keras when it is imported. Import each submodule by its full name.
"""
