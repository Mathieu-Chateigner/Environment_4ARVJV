using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Individual
{
    public GameObject creature;
    public NeuralNetwork neuralNetwork;
    public float fitness;

    public Individual(GameObject c, NeuralNetwork nn)
    {
        creature = c;
        neuralNetwork = nn;
    }
}
