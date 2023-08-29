import numpy as np
from pymanopt.manifolds.stiefel import Stiefel
import matplotlib.pyplot as plt

def plot_on_semicircle(X, colors, title):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    u = np.linspace(6*np.pi/14, 8*np.pi/14, 100)
    xx = np.cos(u)
    yy = np.sin(u)
    ax.plot(xx, yy, alpha=0.1)
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100)
    ax.set_xlabel('Dimension 1', fontsize=16)
    ax.set_ylabel('Dimension 2', fontsize=16)
    plt.title(title, fontsize=20)
    plt.show()

def plot_on_circle(X, colors, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    u = np.linspace(0, 2*np.pi, 100)
    xx = np.cos(u)
    yy = np.sin(u)
    ax.plot(xx, yy, alpha=0.1)
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100)
    ax.set_xlabel('Dimension 1', fontsize=16)
    ax.set_ylabel('Dimension 2', fontsize=16)
    plt.title(title, fontsize=20)
    plt.show()

def plot_on_hemisphere(X, colors, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi/12:20j]
    xx = np.cos(u) * np.sin(v)
    yy = np.sin(u) * np.sin(v)
    zz = np.cos(v)
    ax.plot_surface(xx, yy, zz, alpha=0.1)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=100)
    ax.set_xlabel('Dimension 1', fontsize=16)
    ax.set_ylabel('Dimension 2', fontsize=16)
    ax.set_zlabel('Dimension 3', fontsize=16)
    ax.view_init(elev=90)
    plt.title(title, fontsize=20)
    plt.show()

def plot_on_sphere(X, colors, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0: 2 * np.pi : 100j, 0: np.pi :100j]
    xx = np.cos(u) * np.sin(v)
    yy = np.sin(u) * np.sin(v)
    zz = np.cos(v)
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.1, linewidth=0)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=20)
    ax.set_xlabel('Dimension 1', fontsize=16)
    ax.set_ylabel('Dimension 2', fontsize=16)
    ax.set_zlabel('Dimension 3', fontsize=16)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_box_aspect((1,1,1))
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


