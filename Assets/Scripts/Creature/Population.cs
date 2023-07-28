using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Population
{
    public Individual[] individuals;
    
    public Individual getFittest()
    {
        Individual fittest = individuals[0];
        foreach (Individual individual in individuals)
        {
            if (individual.fitness < fittest.fitness)
            {
                fittest = individual;
            }
        }
        
        return fittest;
    }
}
