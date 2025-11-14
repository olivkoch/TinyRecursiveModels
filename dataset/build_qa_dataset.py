#!/usr/bin/env python3
"""
Build Q&A dataset for TinyRecursiveModels training.
Generates question-answer pairs using natural language sentences.
"""

import json
import numpy as np
import random
from pathlib import Path
import argparse

from common import PuzzleDatasetMetadata

# Ultra-Advanced Q&A templates for sophisticated reasoning tasks
QA_TEMPLATES = [
    # Advanced Mathematics - Algebra and Equations
    {
        "questions": [
            "Solve for x: 2x + 3 = 7. What is x?",
            "If 3x - 5 = 13, what is the value of x?",
            "Solve the equation: 4(x + 2) = 24. What is x?",
            "What is the solution to: x² - 9 = 0?",
            "If f(x) = 2x + 1 and f(3) = ?, what is the answer?",
        ],
        "answers": ["{result}"],
        "data": [
            {"result": "2"},
            {"result": "6"},
            {"result": "4"},
            {"result": "x = 3 or x = -3"},
            {"result": "7"},
        ]
    },
    # Advanced Mathematics - Word Problems with Multiple Steps
    {
        "questions": [
            "A man buys a horse for $60, sells it for $70, buys it back for $80, and sells it again for $90. How much profit did he make?",
            "If a plane can fly 500 miles per hour in still air, and there is a 50 mph headwind, what is the plane's ground speed?",
            "A ladder 13 feet long leans against a wall. The base is 5 feet from the wall. How high up the wall does the ladder reach?",
            "If a car travels at 60 mph for 2 hours and 40 mph for 1 hour, what is the average speed for the entire trip?",
            "A rectangular garden is twice as long as it is wide. If the perimeter is 60 feet, what are the dimensions?",
        ],
        "answers": ["{result}"],
        "data": [
            {"result": "$20 profit"},
            {"result": "450 mph"},
            {"result": "12 feet"},
            {"result": "53.33 mph"},
            {"result": "10 feet by 20 feet"},
        ]
    },
    # Philosophical and Ethical Reasoning
    {
        "questions": [
            "If a trolley is heading toward 5 people and you can switch it to a track with 1 person, should you do it? Why?",
            "Is it ever morally acceptable to lie? Give an example.",
            "What is the difference between justice and fairness?",
            "Can machines ever truly think, or do they just simulate thinking?",
            "What makes an action morally right or wrong?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Yes, because it minimizes harm (utilitarian ethics)"},
            {"answer": "Sometimes, such as to protect someone from harm"},
            {"answer": "Justice is about rules and punishment, fairness is about equality"},
            {"answer": "This is the philosophical zombie problem - we cannot know for certain"},
            {"answer": "According to deontology: following rules; utilitarianism: maximizing happiness"},
        ]
    },
    # Advanced Scientific Concepts
    {
        "questions": [
            "Explain quantum entanglement in simple terms.",
            "What is the uncertainty principle in quantum mechanics?",
            "How does natural selection explain antibiotic resistance in bacteria?",
            "What is the difference between genotype and phenotype?",
            "Explain how CRISPR gene editing works.",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Two particles that remain connected so that the state of one instantly influences the other, regardless of distance"},
            {"answer": "We cannot simultaneously know both the position and momentum of a particle with perfect accuracy"},
            {"answer": "Bacteria with random mutations that make them resistant survive and reproduce when antibiotics are present"},
            {"answer": "Genotype is the genetic code, phenotype is the physical expression of those genes"},
            {"answer": "CRISPR uses guide RNA to target specific DNA sequences, then Cas9 enzyme cuts the DNA for editing"},
        ]
    },
    # Complex Logical Puzzles and Paradoxes
    {
        "questions": [
            "This sentence is false. Is this statement true or false?",
            "Can an omnipotent being create a stone so heavy that even they cannot lift it?",
            "If God is omnipotent, can God create a being more powerful than itself?",
            "What is the unexpected hanging paradox?",
            "Explain the Monty Hall problem and why switching doors improves your chances.",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "This creates a paradox - if true, then it's false; if false, then it's true"},
            {"answer": "This is a paradox that challenges the concept of omnipotence"},
            {"answer": "No, because that would contradict the definition of being the most powerful"},
            {"answer": "A prisoner is told they will be hanged on a surprise date, leading to logical contradictions"},
            {"answer": "With 3 doors and 1 car, switching gives 2/3 chance vs 1/3 for staying"},
        ]
    },
    # Historical Analysis and Inference
    {
        "questions": [
            "Why did the Roman Empire fall? Give three main reasons.",
            "What were the primary causes of World War I?",
            "How did the Industrial Revolution change society?",
            "What was the significance of the Magna Carta?",
            "Why did the Soviet Union collapse?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Economic troubles, military overextension, political corruption, and barbarian invasions"},
            {"answer": "Nationalism, imperialism, militarism, and the assassination of Archduke Franz Ferdinand"},
            {"answer": "Mass production, urbanization, new social classes, and technological advancement"},
            {"answer": "It limited the king's power and established the principle that everyone is subject to the law"},
            {"answer": "Economic stagnation, political repression, the Afghanistan war, and Gorbachev's reforms"},
        ]
    },
    # Advanced Psychology and Human Behavior
    {
        "questions": [
            "What is cognitive dissonance and give an example?",
            "Explain the bystander effect with an example.",
            "What is confirmation bias and how does it affect decision making?",
            "Describe the difference between intrinsic and extrinsic motivation.",
            "What is the Dunning-Kruger effect?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Mental discomfort from holding contradictory beliefs, like smoking while knowing it's unhealthy"},
            {"answer": "People are less likely to help in emergencies when others are present, as seen in the Kitty Genovese case"},
            {"answer": "Tendency to seek information that confirms existing beliefs and ignore contradictory evidence"},
            {"answer": "Intrinsic comes from within (enjoyment), extrinsic comes from external rewards (money, grades)"},
            {"answer": "Less competent people overestimate their abilities while highly competent people underestimate theirs"},
        ]
    },
    # Complex Systems and Economics
    {
        "questions": [
            "Explain how compound interest works with an example.",
            "What is the tragedy of the commons?",
            "How does inflation affect purchasing power?",
            "Explain the concept of supply and demand with an example.",
            "What is opportunity cost?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Interest earned on both principal and accumulated interest - $100 at 10% becomes $110, then $121, etc."},
            {"answer": "Shared resources get overused because individuals prioritize self-interest over collective good"},
            {"answer": "It reduces the value of money, so you need more dollars to buy the same goods"},
            {"answer": "High demand + low supply = high prices; low demand + high supply = low prices"},
            {"answer": "The value of the best alternative you give up when making a choice"},
        ]
    },
    # Advanced Language and Literature Analysis
    {
        "questions": [
            "What is irony and give three types with examples?",
            "Explain the difference between denotation and connotation.",
            "What is a metaphor versus a simile?",
            "Analyze the theme of isolation in Mary Shelley's Frankenstein.",
            "What is stream of consciousness in literature?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Verbal (saying opposite of what you mean), situational (outcome opposite of expectation), dramatic (audience knows more than characters)"},
            {"answer": "Denotation is literal meaning, connotation is emotional/cultural associations"},
            {"answer": "Metaphor directly equates things (time is a thief), simile uses 'like' or 'as' (time like a thief)"},
            {"answer": "The creature and Victor both experience profound isolation, leading to their destructive behaviors"},
            {"answer": "A narrative technique that presents thoughts as they flow through a character's mind"},
        ]
    },
    # Advanced Mathematics - Calculus and Analysis
    {
        "questions": [
            "What is the derivative of f(x) = x³ + 2x² - 5x + 1?",
            "Evaluate the integral ∫(3x² + 2x + 1)dx",
            "What is the limit as x approaches 0 of (sin(x))/x?",
            "Find the critical points of f(x) = x³ - 3x² + 2",
            "What is the fundamental theorem of calculus?",
            "Solve the differential equation dy/dx = 2x + 1",
            "What is the Taylor series expansion of e^x around x=0?",
            "Explain the concept of convergence in infinite series",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "f'(x) = 3x² + 4x - 5"},
            {"answer": "x³ + x² + x + C"},
            {"answer": "1 (this is a fundamental limit in calculus)"},
            {"answer": "x = 1 and x = 2 (local maximum and minimum)"},
            {"answer": "It connects differentiation and integration - the derivative of an integral gives the original function"},
            {"answer": "y = x² + x + C"},
            {"answer": "1 + x + x²/2! + x³/3! + x⁴/4! + ..."},
            {"answer": "A series converges if its partial sums approach a finite limit"},
        ]
    },
    # Theoretical Computer Science
    {
        "questions": [
            "What is P vs NP problem and why is it important?",
            "Explain the halting problem and its implications",
            "What is computational complexity class NP-complete?",
            "How does public-key cryptography work?",
            "What is the difference between deterministic and nondeterministic Turing machines?",
            "Explain the concept of algorithmic information theory",
            "What are the limitations of neural networks in computation?",
            "How does quantum computing differ from classical computing?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "P is problems solvable in polynomial time, NP is verifiable in polynomial time - if P=NP, many cryptography systems would break"},
            {"answer": "No algorithm can determine if an arbitrary program will halt - proves fundamental limits of computation"},
            {"answer": "Problems that are in NP and to which all other NP problems can be reduced - solving one solves all"},
            {"answer": "Uses two keys: public for encryption, private for decryption - based on mathematical trapdoor functions"},
            {"answer": "Deterministic follows one path, nondeterministic can explore multiple paths simultaneously"},
            {"answer": "Studies the information content of algorithms - Kolmogorov complexity measures randomness"},
            {"answer": "They are universal approximators but may not learn efficiently and lack true understanding"},
            {"answer": "Uses quantum superposition and entanglement for parallel computation of multiple states"},
        ]
    },
    # Advanced Physics - Relativity and Quantum Field Theory
    {
        "questions": [
            "Explain Einstein's theory of special relativity in simple terms",
            "What is the twin paradox in relativity?",
            "How does general relativity explain gravity?",
            "What is quantum field theory?",
            "Explain the Higgs mechanism and the Higgs boson",
            "What is the cosmological constant problem?",
            "How does black hole information paradox challenge physics?",
            "What is string theory trying to accomplish?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Time and space are relative, not absolute - simultaneity depends on reference frame, nothing travels faster than light"},
            {"answer": "One twin travels at near light speed and returns younger due to time dilation"},
            {"answer": "Gravity is curvature of spacetime caused by mass-energy, not a force between objects"},
            {"answer": "Framework unifying quantum mechanics and special relativity - particles are excitations in quantum fields"},
            {"answer": "Mechanism that gives particles mass through interaction with Higgs field - Higgs boson is the field excitation"},
            {"answer": "Why is the measured cosmological constant 120 orders of magnitude smaller than quantum predictions?"},
            {"answer": "Black holes destroy information, contradicting quantum mechanics' unitarity principle"},
            {"answer": "Unify all fundamental forces and explain particle properties through vibrating strings in higher dimensions"},
        ]
    },
    # Neuroscience and Cognitive Science
    {
        "questions": [
            "How does synaptic plasticity enable learning?",
            "What is the binding problem in neuroscience?",
            "Explain the difference between working memory and long-term memory",
            "How does the brain's predictive coding work?",
            "What is embodied cognition?",
            "Explain neural oscillations and their cognitive functions",
            "How does the mirror neuron system work?",
            "What is the hard problem of consciousness?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Synapses strengthen or weaken based on correlated activity - Hebbian learning: 'neurons that fire together wire together'"},
            {"answer": "How different sensory features (color, shape, motion) are integrated into coherent object perception"},
            {"answer": "Working memory holds information temporarily for processing, long-term memory stores it permanently"},
            {"answer": "Brain predicts sensory input and learns from prediction errors to minimize surprise"},
            {"answer": "Cognition arises from interactions between brain, body, and environment, not just brain alone"},
            {"answer": "Rhythmic brain activity coordinates information processing across different frequency bands"},
            {"answer": "Neurons fire both when performing actions and observing others perform same actions - enables empathy and imitation"},
            {"answer": "Why subjective experience (qualia) exists - the 'what it's like' aspect of consciousness"},
        ]
    },
    # Advanced Chemistry and Biochemistry
    {
        "questions": [
            "Explain quantum chemistry and molecular orbital theory",
            "How does enzyme catalysis work at the molecular level?",
            "What is the difference between primary, secondary, and tertiary protein structure?",
            "Explain the citric acid cycle and its role in metabolism",
            "How does PCR (polymerase chain reaction) work?",
            "What is the role of ATP in cellular energy transfer?",
            "Explain the mechanism of photosynthesis at the quantum level",
            "How do neurotransmitters cross the synaptic cleft?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Applies quantum mechanics to chemical systems - molecular orbitals form from atomic orbital combinations"},
            {"answer": "Enzymes lower activation energy by stabilizing transition states through specific binding interactions"},
            {"answer": "Primary: amino acid sequence; Secondary: local folding (alpha helices, beta sheets); Tertiary: overall 3D structure"},
            {"answer": "Circular metabolic pathway that oxidizes acetyl-CoA to CO2, producing NADH, FADH2, and ATP"},
            {"answer": "Uses heat-stable polymerase to exponentially amplify DNA through repeated denaturation, annealing, and extension cycles"},
            {"answer": "ATP is the universal energy currency - hydrolysis releases energy for cellular work through phosphate transfer"},
            {"answer": "Quantum coherence allows efficient energy transfer through photosynthetic complexes despite thermal noise"},
            {"answer": "Neurotransmitters diffuse across synapse and bind to receptors on postsynaptic neuron, triggering signal cascades"},
        ]
    },
    # Linguistics and Philosophy of Language
    {
        "questions": [
            "What is the Sapir-Whorf hypothesis?",
            "Explain Chomsky's theory of generative grammar",
            "What is the difference between semantics and pragmatics?",
            "How does speech act theory work?",
            "What is the problem of reference in philosophy of language?",
            "Explain the concept of linguistic relativity",
            "How does language acquisition work in children?",
            "What is the philosophy of meaning in Wittgenstein's later work?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Language determines thought - the structure of language influences how speakers perceive and conceptualize the world"},
            {"answer": "Humans have innate universal grammar that generates all possible grammatical sentences"},
            {"answer": "Semantics studies literal meaning, pragmatics studies meaning in context and speaker intentions"},
            {"answer": "Utterances don't just convey information but perform actions (promising, requesting, apologizing)"},
            {"answer": "How words connect to things in the world - Frege's distinction between sense and reference"},
            {"answer": "Different languages categorize reality differently, affecting cognition and worldview"},
            {"answer": "Children use innate language acquisition device plus environmental input to learn grammar rules"},
            {"answer": "Meaning is use - language games in social contexts determine word meaning, not mental representations"},
        ]
    },
    # Advanced Statistics and Probability Theory
    {
        "questions": [
            "Explain Bayesian inference with an example",
            "What is the central limit theorem and why is it important?",
            "How does maximum likelihood estimation work?",
            "What is the difference between frequentist and Bayesian statistics?",
            "Explain the concept of statistical power",
            "How does bootstrapping work for statistical inference?",
            "What is the curse of dimensionality?",
            "Explain Markov chains and their applications",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Updates beliefs based on evidence - prior probability × likelihood gives posterior probability"},
            {"answer": "Sample means approach normal distribution regardless of population distribution, enabling statistical inference"},
            {"answer": "Finds parameter values that maximize probability of observing the data given the model"},
            {"answer": "Frequentist: long-run frequencies; Bayesian: degree of belief updated with evidence"},
            {"answer": "Probability of correctly rejecting false null hypothesis - higher power means better chance of detecting effects"},
            {"answer": "Resamples data with replacement to estimate sampling distributions without assumptions"},
            {"answer": "As dimensions increase, data becomes sparse, making distance-based algorithms less effective"},
            {"answer": "Systems where future states depend only on current state - used in physics, biology, and algorithms"},
        ]
    },
    # Game Theory and Decision Theory
    {
        "questions": [
            "Explain the prisoner's dilemma",
            "What is Nash equilibrium?",
            "How does prospect theory differ from expected utility theory?",
            "What is the difference between cooperative and non-cooperative game theory?",
            "Explain the concept of zero-sum games",
            "How does evolutionary game theory work?",
            "What is the tragedy of the commons in game theoretic terms?",
            "Explain backward induction in extensive form games",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Two prisoners must choose to confess or stay silent - rational self-interest leads to worse outcome for both"},
            {"answer": "Strategy profile where no player can benefit by unilaterally changing strategy"},
            {"answer": "People value losses more than equivalent gains and make decisions based on reference points"},
            {"answer": "Cooperative allows binding agreements, non-cooperative assumes self-enforcing strategies"},
            {"answer": "One player's gains equal other player's losses - total payoff is constant"},
            {"answer": "Applies game theory to biological evolution - strategies that survive are evolutionarily stable"},
            {"answer": "Multiple players overexploit shared resource because individual benefit exceeds collective cost"},
            {"answer": "Reasoning backward from end of game to determine optimal strategies at each decision point"},
        ]
    },
    # Systems Theory and Complexity Science
    {
        "questions": [
            "What is emergence in complex systems?",
            "Explain the concept of self-organization",
            "How does chaos theory relate to predictability?",
            "What is the difference between complicated and complex systems?",
            "Explain the concept of attractors in dynamical systems",
            "How does network theory explain real-world phenomena?",
            "What is the edge of chaos hypothesis?",
            "Explain the concept of fractal dimensionality",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Higher-level properties arise from interactions of lower-level components that aren't predictable from individual parts"},
            {"answer": "Systems spontaneously form organized structures without external control through local interactions"},
            {"answer": "Small changes in initial conditions can lead to vastly different outcomes - deterministic but unpredictable"},
            {"answer": "Complicated systems are analyzable by breaking into parts, complex systems have irreducible emergent properties"},
            {"answer": "States that systems tend toward over time - points, cycles, or strange attractors"},
            {"answer": "Studies how network structure affects system behavior - scale-free networks, small-world properties"},
            {"answer": "Complex systems are most adaptive and creative at the boundary between order and chaos"},
            {"answer": "Non-integer dimensions that characterize self-similar patterns at different scales"},
        ]
    },
    # Advanced Engineering and Information Theory
    {
        "questions": [
            "How does control theory work in engineering systems?",
            "Explain Shannon's information theory",
            "What is the difference between analog and digital control systems?",
            "How does feedback work in control systems?",
            "What is the Nyquist-Shannon sampling theorem?",
            "Explain the concept of entropy in information theory",
            "How does error-correcting codes work?",
            "What is the difference between open and closed loop control?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Uses mathematical models to design systems that maintain desired outputs despite disturbances"},
            {"answer": "Quantifies information content and communication capacity - entropy measures uncertainty"},
            {"answer": "Analog uses continuous signals, digital uses discrete values - digital is more robust to noise"},
            {"answer": "System output is measured and compared to desired output to generate corrective action"},
            {"answer": "To perfectly reconstruct a signal, sample at least twice the highest frequency component"},
            {"answer": "Measure of uncertainty or information content - higher entropy means more uncertainty"},
            {"answer": "Add redundant bits to detect and correct errors in data transmission"},
            {"answer": "Open loop has no feedback, closed loop uses feedback to adjust output based on measurements"},
        ]
    },
    # Advanced Biology and Evolutionary Theory
    {
        "questions": [
            "Explain epigenetics and its role in evolution",
            "How does neutral theory of evolution differ from natural selection?",
            "What is the extended evolutionary synthesis?",
            "Explain the concept of evolutionary developmental biology (evo-devo)",
            "How does horizontal gene transfer complicate the tree of life?",
            "What is the hologenome theory of evolution?",
            "Explain the concept of niche construction",
            "How does evolutionary game theory apply to social behavior?",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "Heritable changes in gene expression without DNA sequence changes - can influence evolution"},
            {"answer": "Most genetic variation is neutral (no fitness effect) rather than adaptive"},
            {"answer": "Incorporates epigenetics, developmental biology, and ecological inheritance into evolutionary theory"},
            {"answer": "Studies how developmental processes evolve and constrain evolutionary change"},
            {"answer": "Genes can transfer between unrelated species, creating a web rather than tree of relationships"},
            {"answer": "Organisms and their symbiotic microbes co-evolve as a single evolutionary unit"},
            {"answer": "Organisms modify their environment, which then influences their own evolution and that of others"},
            {"answer": "Models social behaviors as evolutionary stable strategies in repeated interactions"},
        ]
    },
    # Philosophy of Science and Epistemology
    {
        "questions": [
            "What is the demarcation problem in philosophy of science?",
            "Explain Popper's falsifiability criterion",
            "How does Kuhn's paradigm shifts work?",
            "What is the problem of induction?",
            "Explain the concept of theory-laden observation",
            "How does Bayesian epistemology work?",
            "What is the difference between justification and truth?",
            "Explain the concept of scientific realism vs anti-realism",
        ],
        "answers": ["{answer}"],
        "data": [
            {"answer": "How to distinguish genuine science from pseudoscience - no universally accepted criterion"},
            {"answer": "Scientific theories must be testable and potentially falsifiable by evidence"},
            {"answer": "Scientific revolutions involve wholesale changes in fundamental frameworks and assumptions"},
            {"answer": "How can we justify believing that future will resemble past based on finite observations?"},
            {"answer": "All observations are interpreted through theoretical frameworks - no theory-neutral facts"},
            {"answer": "Treats belief degrees as probabilities updated by evidence using Bayes' theorem"},
            {"answer": "Justification is about rationally held beliefs, truth is correspondence to reality"},
            {"answer": "Realism: scientific theories describe real entities; Anti-realism: theories are just useful tools"},
        ]
    }
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
    for template in QA_TEMPLATES:
        for data_item in template["data"]:
            for q_template in template["questions"]:
                q = q_template.format(**data_item)
                all_words.update(re.findall(r'\b\w+\b', q.lower()))
            for a_template in template["answers"]:
                a = a_template.format(**data_item)
                all_words.update(re.findall(r'\b\w+\b', a.lower()))

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
        template = random.choice(QA_TEMPLATES)
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
        template = random.choice(QA_TEMPLATES)
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
    parser = argparse.ArgumentParser(description="Build Q&A dataset")
    parser.add_argument("--num-train-puzzles", type=int, default=50000,
                       help="Number of training examples")
    parser.add_argument("--num-test-puzzles", type=int, default=10000,
                       help="Number of test examples")
    parser.add_argument("--output-dir", type=str, default="data/qa_pairs",
                       help="Output directory")

    args = parser.parse_args()

    # Build dataset
    data = build_qa_dataset(args.num_train_puzzles, args.num_test_puzzles)

    # Save to disk
    save_dataset(data, args.output_dir)

if __name__ == "__main__":
    main()