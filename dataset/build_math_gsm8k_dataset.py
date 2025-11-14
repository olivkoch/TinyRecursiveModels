#!/usr/bin/env python3
"""
Build MATH/GSM8K mathematical reasoning dataset for TinyRecursiveModels training.
Generates question-answer pairs for mathematical problem solving.
"""

import json
import numpy as np
import random
from pathlib import Path
import argparse

from common import PuzzleDatasetMetadata

# MATH and GSM8K style mathematical reasoning templates
# MATH and GSM8K style mathematical reasoning templates
MATH_GSM8K_TEMPLATES = [
    # GSM8K Style: Subtraction problems
    {
        "questions": [
            "If {name} has {num1} apples and gives {num2} to {friend}, how many does {name} have left?",
            "{name} started with {num1} stickers and gave away {num2}. How many does {name} have now?",
            "{name} had ${num1} and spent ${num2} on candy. How much money does {name} have left?",
        ],
        "answers": ["{result}"],
        "data": [
            {"name": "Sarah", "num1": "10", "num2": "3", "friend": "John", "result": "7"},
            {"name": "Mike", "num1": "15", "num2": "7", "friend": "Lisa", "result": "8"},
            {"name": "Emma", "num1": "20", "num2": "5", "friend": "Tom", "result": "15"},
        ]
    },
    # GSM8K Style: Multiplication problems
    {
        "questions": [
            "{name} bought {num1} candies. Each candy costs ${price}. How much did {name} spend?",
        ],
        "answers": ["{result}"],
        "data": [
            {"name": "Mike", "num1": "5", "price": "2", "result": "10"},
            {"name": "Lisa", "num1": "4", "price": "3", "result": "12"},
            {"name": "John", "num1": "6", "price": "1", "result": "6"},
        ]
    },
    # GSM8K Style: Division problems
    {
        "questions": [
            "If {num1} people share {total} pizzas equally, how many pizzas does each person get?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num1": "4", "total": "12", "result": "3"},
            {"num1": "3", "total": "15", "result": "5"},
            {"num1": "5", "total": "20", "result": "4"},
        ]
    },
    # GSM8K Style: Addition problems
    {
        "questions": [
            "{name} scored {score1} points in the first game and {score2} in the second. What is the total?",
        ],
        "answers": ["{result}"],
        "data": [
            {"name": "Tom", "score1": "25", "score2": "30", "result": "55"},
            {"name": "Anna", "score1": "20", "score2": "35", "result": "55"},
            {"name": "David", "score1": "15", "score2": "40", "result": "55"},
        ]
    },
    # GSM8K Style: Division problems
    {
        "questions": [
            "If {num1} people share {total} pizzas equally, how many pizzas does each person get?",
            "A train travels {distance} miles in {hours} hours. What is its average speed?",
            "{num1} friends want to share ${total} equally. How much does each friend get?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num1": "4", "total": "12", "distance": "200", "hours": "4", "result": "3"},
            {"num1": "3", "total": "15", "distance": "300", "hours": "5", "result": "5"},
            {"num1": "5", "total": "20", "distance": "250", "hours": "5", "result": "4"},
        ]
    },
    # GSM8K Style: Addition problems
    {
        "questions": [
            "{name} scored {score1} points in the first game and {score2} in the second. What is the total?",
        ],
        "answers": ["{result}"],
        "data": [
            {"name": "Tom", "score1": "25", "score2": "30", "result": "55"},
            {"name": "Anna", "score1": "20", "score2": "35", "result": "55"},
            {"name": "David", "score1": "15", "score2": "40", "result": "55"},
        ]
    },
    # GSM8K Style: Factory Production
    {
        "questions": [
            "A factory produces {rate} toys per hour. How many toys does it produce in {hours} hours?",
        ],
        "answers": ["{result}"],
        "data": [
            {"rate": "50", "hours": "8", "result": "400"},
            {"rate": "25", "hours": "12", "result": "300"},
            {"rate": "40", "hours": "6", "result": "240"},
        ]
    },
    # GSM8K Style: Unit Price
    {
        "questions": [
            "If {num1} books cost ${total}, how much does one book cost?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num1": "3", "total": "15", "result": "5"},
            {"num1": "4", "total": "20", "result": "5"},
            {"num1": "2", "total": "18", "result": "9"},
        ]
    },
    # GSM8K Style: Gift Shopping
    {
        "questions": [
            "{name} has ${money} and wants to buy gifts costing ${cost} each. How many can {name} buy?",
        ],
        "answers": ["{result}"],
        "data": [
            {"name": "Lisa", "money": "50", "cost": "10", "result": "5"},
            {"name": "Tom", "money": "30", "cost": "6", "result": "5"},
            {"name": "Anna", "money": "45", "cost": "9", "result": "5"},
        ]
    },
    # GSM8K Style: Garden Area
    {
        "questions": [
            "A garden is {length} feet long and {width} feet wide. What is its area?",
        ],
        "answers": ["{result}"],
        "data": [
            {"length": "20", "width": "15", "result": "300 square feet"},
            {"length": "12", "width": "8", "result": "96 square feet"},
            {"length": "25", "width": "10", "result": "250 square feet"},
        ]
    },
    # GSM8K Style: Student Pencils
    {
        "questions": [
            "If {num1} students each bring {num2} pencils, how many pencils are there total?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num1": "6", "num2": "4", "result": "24"},
            {"num1": "5", "num2": "3", "result": "15"},
            {"num1": "8", "num2": "2", "result": "16"},
        ]
    },
    # MATH Style: Linear Equations
    {
        "questions": [
            "Solve for x: {eq}. What is x?",
        ],
        "answers": ["{result}"],
        "data": [
            {"eq": "2x + 5 = 13", "result": "4"},
            {"eq": "3x - 7 = 8", "result": "5"},
            {"eq": "4x + 2 = 18", "result": "4"},
        ]
    },
    # MATH Style: System of Equations
    {
        "questions": [
            "If {eq1} and {eq2}, what is the value of x?",
        ],
        "answers": ["{result}"],
        "data": [
            {"eq1": "3x + 2 = 11", "eq2": "x - 1 = 3", "result": "4"},
            {"eq1": "2x + y = 10", "eq2": "x - y = 2", "result": "3"},
            {"eq1": "x + 2y = 8", "eq2": "2x - y = 1", "result": "2"},
        ]
    },
    # MATH Style: Quadratic Equations
    {
        "questions": [
            "Find x such that {eq} = 0",
        ],
        "answers": ["{result}"],
        "data": [
            {"eq": "x² - 9", "result": "x = 3 or x = -3"},
            {"eq": "x² - 4", "result": "x = 2 or x = -2"},
            {"eq": "2x² - 8", "result": "x = 2 or x = -2"},
        ]
    },
    # MATH Style: Circle Area
    {
        "questions": [
            "What is the area of a circle with radius {r}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"r": "5", "result": "25π"},
            {"r": "3", "result": "9π"},
            {"r": "7", "result": "49π"},
        ]
    },
    # MATH Style: Triangle Area
    {
        "questions": [
            "A triangle has base {base} and height {height}. What is its area?",
        ],
        "answers": ["{result}"],
        "data": [
            {"base": "10", "height": "8", "result": "40"},
            {"base": "6", "height": "4", "result": "12"},
            {"base": "15", "height": "12", "result": "90"},
        ]
    },
    # MATH Style: Sphere Volume
    {
        "questions": [
            "Find the volume of a sphere with radius {r}",
        ],
        "answers": ["{result}"],
        "data": [
            {"r": "3", "result": "36π"},
            {"r": "2", "result": "32π/3"},
            {"r": "4", "result": "256π/3"},
        ]
    },
    # MATH Style: Circle Circumference
    {
        "questions": [
            "What is the circumference of a circle with diameter {d}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"d": "10", "result": "10π"},
            {"d": "6", "result": "6π"},
            {"d": "14", "result": "14π"},
        ]
    },
    # MATH Style: Rectangle Perimeter
    {
        "questions": [
            "A rectangle is {l} by {w}. What is its perimeter?",
        ],
        "answers": ["{result}"],
        "data": [
            {"l": "12", "w": "8", "result": "40"},
            {"l": "5", "w": "3", "result": "16"},
            {"l": "9", "w": "7", "result": "32"},
        ]
    },
    # MATH Style: Derivatives
    {
        "questions": [
            "What is the derivative of {func}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "x²", "result": "2x"},
            {"func": "x³", "result": "3x²"},
            {"func": "sin(x)", "result": "cos(x)"},
        ]
    },
    # MATH Style: Indefinite Integrals
    {
        "questions": [
            "Find the integral of {func} dx",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "3x² + 2x", "result": "x³ + x² + C"},
            {"func": "x²", "result": "x³/3 + C"},
            {"func": "e^x", "result": "e^x + C"},
        ]
    },
    # MATH Style: Definite Integrals
    {
        "questions": [
            "Evaluate ∫_{a}^{b} {func} dx",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "2x", "a": "0", "b": "1", "result": "1"},
            {"func": "x", "a": "0", "b": "2", "result": "2"},
            {"func": "3x²", "a": "1", "b": "2", "result": "7"},
        ]
    },
    # MATH Style: System of Linear Equations
    {
        "questions": [
            "Solve the system: {eq1}, {eq2}",
        ],
        "answers": ["{result}"],
        "data": [
            {"eq1": "2x + 3y = 7", "eq2": "x - y = 1", "result": "x=2, y=1"},
            {"eq1": "x + y = 5", "eq2": "2x - y = 1", "result": "x=2, y=3"},
            {"eq1": "3x + 2y = 8", "eq2": "x - y = 1", "result": "x=2, y=1"},
        ]
    },
    # MATH Style: Matrix Determinant
    {
        "questions": [
            "Find the determinant of [{a},{b};{c},{d}]",
        ],
        "answers": ["{result}"],
        "data": [
            {"a": "1", "b": "2", "c": "3", "d": "4", "result": "-2"},
            {"a": "2", "b": "1", "c": "1", "d": "3", "result": "5"},
            {"a": "3", "b": "0", "c": "0", "d": "2", "result": "6"},
        ]
    },
    # MATH Style: Matrix Inverse
    {
        "questions": [
            "What is the inverse of [{p},{q};{r},{s}]?",
        ],
        "answers": ["{result}"],
        "data": [
            {"p": "1", "q": "2", "r": "3", "s": "4", "result": "[-2,1;1.5,-0.5]"},
            {"p": "2", "q": "1", "r": "1", "s": "1", "result": "[1/3,-1/3;-1/3,2/3]"},
            {"p": "1", "q": "0", "r": "0", "s": "2", "result": "[1,0;0,0.5]"},
        ]
    },
    # MATH Style: Probability - Independent Events
    {
        "questions": [
            "If P(A) = {p_a} and P(B) = {p_b}, what is P(A and B) if independent?",
        ],
        "answers": ["{result}"],
        "data": [
            {"p_a": "0.3", "p_b": "0.4", "result": "0.12"},
            {"p_a": "0.2", "p_b": "0.5", "result": "0.1"},
            {"p_a": "0.4", "p_b": "0.3", "result": "0.12"},
        ]
    },
    # MATH Style: Statistics - Mean
    {
        "questions": [
            "What is the mean of {nums}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"nums": "1,2,3,4,5", "result": "3"},
            {"nums": "2,4,6,8", "result": "5"},
            {"nums": "1,3,5,7,9", "result": "5"},
        ]
    },
    # MATH Style: Statistics - Standard Deviation
    {
        "questions": [
            "Find the standard deviation of a set with variance {var}",
        ],
        "answers": ["{result}"],
        "data": [
            {"var": "4", "result": "2"},
            {"var": "9", "result": "3"},
            {"var": "16", "result": "4"},
        ]
    },
    # MATH Style: Binomial Probability
    {
        "questions": [
            "In a binomial experiment with n={n}, p={p}, what is P(X={k})?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "10", "p": "0.5", "k": "5", "result": "0.246"},
            {"n": "5", "p": "0.3", "k": "2", "result": "0.309"},
            {"n": "8", "p": "0.25", "k": "3", "result": "0.207"},
        ]
    },
    # MATH Style: Die Probability
    {
        "questions": [
            "What is the probability of rolling a {num} on a fair die?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num": "6", "result": "1/6"},
            {"num": "1", "result": "1/6"},
            {"num": "4", "result": "1/6"},
        ]
    },
    # MATH Style: GCD Problems
    {
        "questions": [
            "What is gcd({a}, {b})?",
        ],
        "answers": ["{result}"],
        "data": [
            {"a": "24", "b": "36", "result": "12"},
            {"a": "15", "b": "25", "result": "5"},
            {"a": "18", "b": "30", "result": "6"},
        ]
    },
    # MATH Style: Prime Number Check
    {
        "questions": [
            "Is {num} a prime number?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num": "17", "result": "Yes"},
            {"num": "15", "result": "No"},
            {"num": "23", "result": "Yes"},
        ]
    },
    # MATH Style: Modular Arithmetic
    {
        "questions": [
            "Solve {a}x ≡ {b} mod {m}",
        ],
        "answers": ["{result}"],
        "data": [
            {"a": "2", "b": "4", "m": "6", "result": "x ≡ 2 mod 3"},
            {"a": "3", "b": "2", "m": "5", "result": "x ≡ 4 mod 5"},
            {"a": "1", "b": "3", "m": "7", "result": "x ≡ 3 mod 7"},
        ]
    },
    # MATH Style: Euler's Totient
    {
        "questions": [
            "What is φ({n}) (Euler's totient function)?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "10", "result": "4"},
            {"n": "12", "result": "4"},
            {"n": "15", "result": "8"},
        ]
    },
    # MATH Style: LCM Problems
    {
        "questions": [
            "Find the least common multiple of {a} and {b}",
        ],
        "answers": ["{result}"],
        "data": [
            {"a": "12", "b": "18", "result": "36"},
            {"a": "8", "b": "12", "result": "24"},
            {"a": "15", "b": "20", "result": "60"},
        ]
    },
    # MATH Style: Trigonometric Identities
    {
        "questions": [
            "If sin(θ) = {val} and θ is acute, what is cos(θ)?",
        ],
        "answers": ["{result}"],
        "data": [
            {"val": "3/5", "result": "4/5"},
            {"val": "4/5", "result": "3/5"},
            {"val": "1/2", "result": "√3/2"},
        ]
    },
    # MATH Style: Trigonometric Equations
    {
        "questions": [
            "Solve sin(x) = 0 for x in [0, 2π]",
        ],
        "answers": ["{result}"],
        "data": [
            {"result": "x = 0, π, 2π"},
            {"result": "x = π/2, 3π/2"},
            {"result": "x = π/4, 5π/4, 3π/4, 7π/4"},
        ]
    },
    # MATH Style: Special Angles
    {
        "questions": [
            "What is tan({angle}°)?",
        ],
        "answers": ["{result}"],
        "data": [
            {"angle": "45", "result": "1"},
            {"angle": "30", "result": "√3/3"},
            {"angle": "60", "result": "√3"},
        ]
    },
    # MATH Style: Exact Values
    {
        "questions": [
            "Find the exact value of cos(π/4)",
        ],
        "answers": ["{result}"],
        "data": [
            {"result": "√2/2"},
            {"result": "1/2"},
            {"result": "0"},
        ]
    },
    # MATH Style: Limits
    {
        "questions": [
            "What is the limit of {expr} as x approaches {val}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"expr": "(x²-1)/(x-1)", "val": "1", "result": "2"},
            {"expr": "sin(x)/x", "val": "0", "result": "1"},
            {"expr": "(1-cos(x))/x²", "val": "0", "result": "1/2"},
        ]
    },
    # MATH Style: Taylor Series
    {
        "questions": [
            "Find the Taylor series of {func} around x={a}",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "e^x", "a": "0", "result": "1 + x + x²/2! + x³/3! + ..."},
            {"func": "sin(x)", "a": "0", "result": "x - x³/3! + x⁵/5! - ..."},
            {"func": "cos(x)", "a": "0", "result": "1 - x²/2! + x⁴/4! - ..."},
        ]
    },
    # MATH Style: Higher Order Derivatives
    {
        "questions": [
            "What is the derivative of order {n} of {func}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "2", "func": "sin(x)", "result": "-sin(x)"},
            {"n": "3", "func": "cos(x)", "result": "-cos(x)"},
            {"n": "4", "func": "e^x", "result": "e^x"},
        ]
    },
    # MATH Style: Differential Equations
    {
        "questions": [
            "Solve the differential equation: {eq}",
        ],
        "answers": ["{result}"],
        "data": [
            {"eq": "y' + y = 0", "result": "y = Ce^{-x}"},
            {"eq": "y'' + y = 0", "result": "y = A cos(x) + B sin(x)"},
            {"eq": "y' = ky", "result": "y = Ce^{kx}"},
        ]
    },
    # MATH Style: Permutations
    {
        "questions": [
            "How many permutations of {n} distinct items?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "5", "result": "120"},
            {"n": "4", "result": "24"},
            {"n": "3", "result": "6"},
        ]
    },
    # MATH Style: Combinations
    {
        "questions": [
            "What is C({n}, {k})?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "5", "k": "3", "result": "10"},
            {"n": "6", "k": "2", "result": "15"},
            {"n": "4", "k": "2", "result": "6"},
        ]
    },
    # MATH Style: Recurrence Relations
    {
        "questions": [
            "Solve the recurrence: a_n = {a}*a_(n-1) + {b}*a_(n-2), with a_0={a0}, a_1={a1}, find a_5",
        ],
        "answers": ["{result}"],
        "data": [
            {"a": "1", "b": "1", "a0": "0", "a1": "1", "result": "5"},
            {"a": "2", "b": "1", "a0": "1", "a1": "1", "result": "11"},
            {"a": "1", "b": "2", "a0": "0", "a1": "1", "result": "3"},
        ]
    },
    # MATH Style: Permutations with Repetition
    {
        "questions": [
            "How many ways to choose {k} items from {n} with order?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "5", "k": "3", "result": "60"},
            {"n": "4", "k": "2", "result": "12"},
            {"n": "6", "k": "4", "result": "360"},
        ]
    },
    # MATH Style: Power Sets
    {
        "questions": [
            "What is the number of subsets of a set with {n} elements?",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "3", "result": "8"},
            {"n": "4", "result": "16"},
            {"n": "5", "result": "32"},
        ]
    },
    # MATH Style: Complex Numbers Basics
    {
        "questions": [
            "What is i²?",
        ],
        "answers": ["{result}"],
        "data": [
            {"result": "-1"},
            {"result": "-1"},
            {"result": "-1"},
        ]
    },
    # MATH Style: Complex Roots
    {
        "questions": [
            "Find the roots of z² + {c} = 0",
        ],
        "answers": ["{result}"],
        "data": [
            {"c": "1", "result": "z = i, z = -i"},
            {"c": "-1", "result": "z = 1, z = -1"},
            {"c": "4", "result": "z = 2i, z = -2i"},
        ]
    },
    # MATH Style: Complex Derivatives
    {
        "questions": [
            "What is the derivative of {func}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "e^z", "result": "e^z"},
            {"func": "sin(z)", "result": "cos(z)"},
            {"func": "z²", "result": "2z"},
        ]
    },
    # MATH Style: Contour Integration
    {
        "questions": [
            "Evaluate the contour integral ∮ {func} dz over |z|={r}",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "1/z", "r": "1", "result": "2πi"},
            {"func": "1/z²", "r": "1", "result": "0"},
            {"func": "z", "r": "1", "result": "0"},
        ]
    },
    # MATH Style: Residues
    {
        "questions": [
            "Find the residue of {func} at z={z0}",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "1/(z-1)", "z0": "1", "result": "1"},
            {"func": "1/z", "z0": "0", "result": "1"},
            {"func": "e^z/z", "z0": "0", "result": "1"},
        ]
    },
    # MATH Style: Group Theory - Element Order
    {
        "questions": [
            "What is the order of the element {g} in ℤ/{n}ℤ?",
        ],
        "answers": ["{result}"],
        "data": [
            {"g": "3", "n": "7", "result": "order 3"},
            {"g": "2", "n": "5", "result": "order 4"},
            {"g": "1", "n": "8", "result": "order 1"},
        ]
    },
    # MATH Style: Ring Theory - Field Check
    {
        "questions": [
            "Is {ring} a field?",
        ],
        "answers": ["{result}"],
        "data": [
            {"ring": "ℤ/5ℤ", "result": "Yes"},
            {"ring": "ℤ/4ℤ", "result": "No"},
            {"ring": "ℚ", "result": "Yes"},
        ]
    },
    # MATH Style: Group Homomorphisms
    {
        "questions": [
            "Find the kernel of the homomorphism φ: ℤ → ℤ_{n} given by φ(k) = {mod}",
        ],
        "answers": ["{result}"],
        "data": [
            {"n": "6", "mod": "k mod 6", "result": "6ℤ"},
            {"n": "4", "mod": "k mod 4", "result": "4ℤ"},
            {"n": "8", "mod": "k mod 8", "result": "8ℤ"},
        ]
    },
    # MATH Style: Group Index
    {
        "questions": [
            "What is the index of H in G where |G|={g}, |H|={h}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"g": "12", "h": "4", "result": "3"},
            {"g": "20", "h": "5", "result": "4"},
            {"g": "24", "h": "8", "result": "3"},
        ]
    },
    # MATH Style: Continuity
    {
        "questions": [
            "Is the function f(x) = {func} continuous at x={a}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "x²", "a": "0", "result": "Yes"},
            {"func": "1/x", "a": "0", "result": "No"},
            {"func": "sin(x)", "a": "π", "result": "Yes"},
        ]
    },
    # MATH Style: Series Convergence
    {
        "questions": [
            "Does the series ∑ {term} converge?",
        ],
        "answers": ["{result}"],
        "data": [
            {"term": "1/n²", "result": "Yes (p-series with p=2 > 1)"},
            {"term": "1/n", "result": "No (harmonic series)"},
            {"term": "2⁻ⁿ", "result": "Yes (geometric with r=1/2)"},
        ]
    },
    # MATH Style: Riemann Integration
    {
        "questions": [
            "What is the Riemann integral of {func} from {a} to {b}?",
        ],
        "answers": ["{result}"],
        "data": [
            {"func": "x", "a": "0", "b": "1", "result": "1/2"},
            {"func": "x²", "a": "0", "b": "2", "result": "8/3"},
            {"func": "sin(x)", "a": "0", "b": "π", "result": "2"},
        ]
    },
]

def generate_qa_pair(template, data_item):
    """Generate a single Q&A pair from template and data."""
    question_template = random.choice(template["questions"])
    answer_template = random.choice(template["answers"])

    # Format question and answer
    question = question_template.format(**data_item)
    answer = answer_template.format(**data_item)

    return question, answer

def create_qa_sequence(question, answer, vocab_size=1000):
    """Convert Q&A pair to sequence format for TRM training."""
    # Simple tokenization - split into words and map to token IDs
    # In practice, you'd want a proper tokenizer, but this works for demo

    # Combine question and answer with special tokens
    full_text = f"Question: {question} Answer: {answer}"

    # Simple word-level tokenization (lowercase, basic punctuation)
    import re
    words = re.findall(r'\b\w+\b', full_text.lower())

    # Create a simple vocabulary mapping (in practice, use a real tokenizer)
    vocab = {}
    token_id = 1  # Start from 1, reserve 0 for padding

    # Build vocabulary from all possible words in our templates
    all_words = set()
    for template in MATH_GSM8K_TEMPLATES:
        for data_item in template["data"]:
            for q_template in template["questions"]:
                try:
                    q = q_template.format(**data_item)
                    all_words.update(re.findall(r'\b\w+\b', q.lower()))
                except KeyError:
                    # Skip if formatting fails due to missing keys
                    pass
            for a_template in template["answers"]:
                try:
                    a = a_template.format(**data_item)
                    all_words.update(re.findall(r'\b\w+\b', a.lower()))
                except KeyError:
                    # Skip if formatting fails due to missing keys
                    pass

    # Sort for consistent ordering
    all_words = sorted(list(all_words))
    for word in all_words:
        vocab[word] = token_id
        token_id += 1

    # Convert words to token IDs
    tokens = [vocab.get(word, 1) for word in words]  # Default to 1 for unknown words

    # Pad or truncate to fixed length (adjust as needed)
    seq_len = 32
    if len(tokens) < seq_len:
        tokens.extend([0] * (seq_len - len(tokens)))
    else:
        tokens = tokens[:seq_len]

    return tokens, len(vocab) + 1  # +1 for padding token

def build_qa_dataset(num_train_puzzles=50000, num_test_puzzles=10000):
    """Build Q&A dataset with specified number of examples."""

    print(f"Building Q&A dataset with {num_train_puzzles} training and {num_test_puzzles} test examples...")

    # Generate training data
    train_data = []
    train_inputs = []
    train_labels = []
    train_puzzle_identifiers = []
    train_puzzle_indices = []
    train_group_indices = []

    train_puzzle_indices.append(0)  # Start with 0
    train_group_indices.append(0)   # Start with 0

    puzzle_id = 0

    puzzle_id = 0
    vocab_size = 1000  # Will be updated based on actual vocabulary

    for i in range(num_train_puzzles):
        # Select random template and data item
        template = random.choice(MATH_GSM8K_TEMPLATES)
        data_item = random.choice(template["data"])

        # Generate Q&A pair
        question, answer = generate_qa_pair(template, data_item)

        # Convert to sequence format
        tokens, actual_vocab_size = create_qa_sequence(question, answer, vocab_size)
        vocab_size = max(vocab_size, actual_vocab_size)

        # Create training example (for sequence prediction)
        # Input: question tokens, Label: answer tokens shifted by 1
        input_tokens = tokens[:-1]  # All but last token
        label_tokens = tokens[1:]   # All but first token, shifted

        train_inputs.append(input_tokens)
        train_labels.append(label_tokens)
        train_puzzle_identifiers.append(0)  # Single puzzle type
        puzzle_id += 1
        train_puzzle_indices.append(puzzle_id)
        train_group_indices.append(puzzle_id)

    # Generate test data (same process)
    test_data = []
    test_inputs = []
    test_labels = []
    test_puzzle_identifiers = []
    test_puzzle_indices = []
    test_group_indices = []

    test_puzzle_indices.append(0)  # Start with 0
    test_group_indices.append(0)   # Start with 0

    puzzle_id = 0  # Reset for test

    for i in range(num_test_puzzles):
        template = random.choice(MATH_GSM8K_TEMPLATES)
        data_item = random.choice(template["data"])

        question, answer = generate_qa_pair(template, data_item)
        tokens, _ = create_qa_sequence(question, answer, vocab_size)

        input_tokens = tokens[:-1]
        label_tokens = tokens[1:]

        test_inputs.append(input_tokens)
        test_labels.append(label_tokens)
        test_puzzle_identifiers.append(0)
        puzzle_id += 1
        test_puzzle_indices.append(puzzle_id)
        test_group_indices.append(puzzle_id)

    # Create metadata
    metadata = {
        "vocab_size": vocab_size,
        "num_puzzle_identifiers": 1,
        "seq_len": 31,  # input length (32 - 1)
        "num_train_puzzles": num_train_puzzles,
        "num_test_puzzles": num_test_puzzles,
        "puzzle_type": "qa_pairs",
        "description": "Question-Answer pairs dataset for language modeling"
    }

    return {
        "train": {
            "inputs": train_inputs,
            "labels": train_labels,
            "puzzle_identifiers": train_puzzle_identifiers,
            "puzzle_indices": train_puzzle_indices,
            "group_indices": train_group_indices
        },
        "test": {
            "inputs": test_inputs,
            "labels": test_labels,
            "puzzle_identifiers": test_puzzle_identifiers,
            "puzzle_indices": test_puzzle_indices,
            "group_indices": test_group_indices
        },
        "metadata": metadata
    }

def save_dataset(data, output_dir="data/qa_pairs"):
    """Save dataset to disk in the expected format."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create train and test directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Save JSON data
    with open(output_path / "train.json", "w") as f:
        json.dump(data["train"], f, indent=2)

    with open(output_path / "test.json", "w") as f:
        json.dump(data["test"], f, indent=2)

    with open(output_path / "metadata.json", "w") as f:
        json.dump(data["metadata"], f, indent=2)

    # Convert to numpy arrays and save
    print("Converting to numpy arrays...")

    # Train data
    np.save(train_dir / "all__inputs.npy",
            np.array(data["train"]["inputs"]))
    np.save(train_dir / "all__labels.npy",
            np.array(data["train"]["labels"]))
    np.save(train_dir / "all__puzzle_identifiers.npy",
            np.array(data["train"]["puzzle_identifiers"]))
    np.save(train_dir / "all__puzzle_indices.npy",
            np.array(data["train"]["puzzle_indices"]))
    np.save(train_dir / "all__group_indices.npy",
            np.array(data["train"]["group_indices"]))

    # Test data
    np.save(test_dir / "all__inputs.npy",
            np.array(data["test"]["inputs"]))
    np.save(test_dir / "all__labels.npy",
            np.array(data["test"]["labels"]))
    np.save(test_dir / "all__puzzle_identifiers.npy",
            np.array(data["test"]["puzzle_identifiers"]))
    np.save(test_dir / "all__puzzle_indices.npy",
            np.array(data["test"]["puzzle_indices"]))
    np.save(test_dir / "all__group_indices.npy",
            np.array(data["test"]["group_indices"]))

    # Create and save dataset.json for train
    train_metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=data["metadata"]["vocab_size"],
        seq_len=data["metadata"]["seq_len"],
        num_puzzle_identifiers=1,
        total_groups=data["metadata"]["num_train_puzzles"],
        mean_puzzle_examples=1.0,
        total_puzzles=data["metadata"]["num_train_puzzles"],
        sets=["all"]
    )
    with open(train_dir / "dataset.json", "w") as f:
        json.dump(train_metadata.model_dump(), f)

    # Create and save dataset.json for test
    test_metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=data["metadata"]["vocab_size"],
        seq_len=data["metadata"]["seq_len"],
        num_puzzle_identifiers=1,
        total_groups=data["metadata"]["num_test_puzzles"],
        mean_puzzle_examples=1.0,
        total_puzzles=data["metadata"]["num_test_puzzles"],
        sets=["all"]
    )
    with open(test_dir / "dataset.json", "w") as f:
        json.dump(test_metadata.model_dump(), f)

    print(f"Dataset saved to {output_path}")
    print(f"Vocabulary size: {data['metadata']['vocab_size']}")
    print(f"Training examples: {data['metadata']['num_train_puzzles']}")
    print(f"Test examples: {data['metadata']['num_test_puzzles']}")

def main():
    parser = argparse.ArgumentParser(description="Build MATH/GSM8K Q&A dataset")
    parser.add_argument("--num-train-puzzles", type=int, default=10000,
                       help="Number of training examples")
    parser.add_argument("--num-test-puzzles", type=int, default=2000,
                       help="Number of test examples")
    parser.add_argument("--output-dir", type=str, default="data/math_gsm8k_qa",
                       help="Output directory")

    args = parser.parse_args()

    # Build dataset
    data = build_qa_dataset(args.num_train_puzzles, args.num_test_puzzles)

    # Save to disk
    save_dataset(data, args.output_dir)

if __name__ == "__main__":
    main()