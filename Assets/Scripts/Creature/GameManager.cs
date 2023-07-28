using System;
using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using Core;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using Random = UnityEngine.Random;

public class GameManager : MonoBehaviour
{
    public static GameManager instance;
    private WaveFunctionCollapseGenerator WFCG;

    [SerializeField] private GameObject cubePrefab;

    [SerializeField] private GameObject creature;

    [SerializeField] private int cubesLimit = 20;
    public int nombreDeCubes = 10;
    public float tailleMin = 0.1f; // Taille minimale d'un cube
    public float tailleMax = 0.5f; // Taille maximale d'un cube
    public Color couleur = Color.white;
    
    private bool trainLoop = true;
    private bool executionLoop = false;

    public GameObject[] legs;
    public GameObject body;
    private NeuralNetwork net;

    public float trainDuration = 5f;
    private float currentTrainTime = 0f;
    private float cumulatedTime = 0f;
    public GameObject creaturePrefab;
    public Vector3 spawnPoint = new Vector3(0, 0, 0);
    private Vector3 bodySpawnPoint;
    
    public TextMeshProUGUI status;
    public TextMeshProUGUI generationText;
    public TextMeshProUGUI bestFitnessText;
    public TextMeshProUGUI bestPopulationFitnessText;
    public TextMeshProUGUI executionFitnessText;
    private int generation = 1;
    public bool symetric = false;
    public float trainingSpeed = 2f;
    public float legSpeed = 2f;

    // Genetic algorithm
    private Individual[] population;
    public int popSize;
    public int tailleTournoi;
    public bool elitisme;
    public float mutationRate = 0.1f;
    public float mutationRange = .5f;
    public int hiddenLayers = 4;
    private Individual fittestOfAll = null;

    private bool started = false;
    public Camera camera;
    private float generatingTime = 0f;

    private void Awake()
    {
        // If there is an instance, and it's not me, delete myself.

        if (instance != null && instance != this)
        {
            Destroy(this);
        }
        else
        {
            instance = this;
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        WFCG = WaveFunctionCollapseGenerator.instance;
        bodySpawnPoint = new Vector3(spawnPoint.x, spawnPoint.y + 1.646601f, spawnPoint.z);

        setGeneratingView();
        /*int[] layers = new int[3]{ 5, 5, 4 };
        string[] activation = new string[2] { "sigmoid", "sigmoid" };
        this.net = new NeuralNetwork(layers, activation);
        
        // evaluate individual
        for (int i = 0; i < 20000; i++)
        {
            net.BackPropagate(new float[] { 1, 0, 0, 1, 0.175f },new float[] { 1, 1, 1, 0 }); // marcher si debout
            net.BackPropagate(new float[] { 0, 1, 1, 0, 0.175f },new float[] { 1, 0, 0, 1 }); // marcher si debout
            net.BackPropagate(new float[] { 0, 0, 0, 0, 0.075f },new float[] { 1, 1, 1, 1 }); // relever
            net.BackPropagate(new float[] { 1, 1, 1, 1, 0.075f },new float[] { 0, 0, 0, 0 }); // relever
        }

        net = GenerateRandomNN();
        print("cost: "+ net.cost);*/
    }

    // Update is called once per frame
    void Update()
    {
        // Generating timer
        if (!started)
        {
            generatingTime += Time.deltaTime;
            executionFitnessText.text = "GENERATING TIME : " + generatingTime.ToString("F3") + "s";
        }

        if (!started && WFCG.finishGeneration)
        {
            started = true;
            
            population = InitiatePopulation(popSize, true);
            ColorizePopulation(population);
            Time.timeScale = trainingSpeed;
            
            setTrainingView();
        }

        if (!started) return;
        
        if (Input.GetKeyDown(KeyCode.F))
        {
            Time.timeScale = 1f;
            trainLoop = false;
            executionLoop = true;
            DestroyCreatures(population);
            population = new []{ fittestOfAll };
            RegenerateCreatures(population);
            
            setExecutingView();
        }

        if (trainLoop)
        {
            currentTrainTime += Time.deltaTime;
            if (currentTrainTime > trainDuration)
            {
                currentTrainTime = 0f;
                generation++;
                generationText.text = "GEN " + generation;
                CalculateFitness(population);
                bestPopulationFitnessText.text = "BEST POPULATION FITNESS : " + getFittest(population).fitness;
                
                if (fittestOfAll == null) fittestOfAll = getFittest(population);
                if (getFittest(population).fitness < fittestOfAll.fitness) {
                    fittestOfAll = new Individual(null, getFittest(population).neuralNetwork);
                    fittestOfAll.fitness = getFittest(population).fitness;
                    bestFitnessText.text = "BEST OF ALL FITNESS : " + fittestOfAll.fitness;
                }

                DestroyCreatures(population);
                population = EvolvePopulation(population);
                RegenerateCreatures(population);
                trainDuration += .2f;
            }
            runNN3(population);

            camera.orthographicSize = 3;
            camera.transform.eulerAngles = new Vector3(25f, 45f, camera.transform.eulerAngles.z);
            camera.transform.position = new Vector3(population[0].creature.transform.GetChild(0).position.x - 1f, 1f + population[0].creature.transform.GetChild(0).position.y, population[0].creature.transform.GetChild(0).position.z - 1f);
        }
        
        if (executionLoop)
        {
            runNN3(population);
            executionFitnessText.text = "CURRENT FITNESS : " +  (fittestOfAll.fitness = 1 / Vector3.Distance(bodySpawnPoint, fittestOfAll.creature.transform.GetChild(0).position));;
            
            camera.orthographicSize = 2;
            camera.transform.eulerAngles = new Vector3(25f, 45, camera.transform.eulerAngles.z);
            camera.transform.position = new Vector3(population[0].creature.transform.GetChild(0).position.x - 1f, 1f + population[0].creature.transform.GetChild(0).position.y, population[0].creature.transform.GetChild(0).position.z - 1f);
            //Debug.Log(population[0].creature.transform.GetChild(0).position.z);
        }

        cumulatedTime += Time.deltaTime;
    }
    
    private Individual[] InitiatePopulation(int size, bool init=false)
    {
        Individual[] newPopulation = new Individual[size];
        for (int i = 0; i < size; i++)
        {
            GameObject c = null;
            if (init) c = Instantiate(creaturePrefab, spawnPoint, Quaternion.Euler(0f, 0f, 0f));
            
            newPopulation[i] = new Individual(c, GenerateRandomNN());
        }

        return newPopulation;
    }
    
    public Individual[] EvolvePopulation(Individual[] pop)
    {
        Individual[] newPop = InitiatePopulation(popSize);
        int elitismeOffset = 0;
        if (elitisme && popSize > 5)
        {
            newPop[0] = getFittest(pop);
            newPop[1] = getFittest(pop);
            newPop[2] = getFittest(pop);
            newPop[3] = getFittest(pop);
            newPop[4] = getFittest(pop);
            elitismeOffset = 5;
        }
        
        // Selection et crossover
        for (int i = elitismeOffset; i < newPop.Length; i++)
        {
            Individual parent1 = SelectionTournoi(pop);
            Individual parent2 = SelectionTournoi(pop);

            NeuralNetwork childNeuralNetwork = parent1.neuralNetwork.Crossover(parent2.neuralNetwork);
            Individual enfant = new Individual(null, childNeuralNetwork);
            newPop[i] = enfant;
        }
        
        // Mutation
        for (int i = elitismeOffset; i < newPop.Length; i++)
        {
            newPop[i].neuralNetwork.Mutate(mutationRate, mutationRange);
        }
        
        return newPop;
    }
    
    public void CalculateFitness(Individual[] individuals)
    {
        foreach (Individual individual in individuals)
        {
            individual.fitness = 1 / Vector3.Distance(new Vector3(0f, 1.646601f, 0f), individual.creature.transform.GetChild(0).position);
        }
    }
    
    public Individual getFittest(Individual[] individuals)
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
    
    private Individual SelectionTournoi(Individual[] individuals)
    {
        Individual[] tournoi = new Individual[tailleTournoi];
        
        for (int i = 0; i < tailleTournoi; i++)
        {
            int randomIndex = Random.Range(0, individuals.Length);
            tournoi[i] = individuals[randomIndex];
        }

        return getFittest(tournoi);
    }

    private void DestroyCreatures(Individual[] individuals)
    {
        for (int i = 0; i < individuals.Length; i++)
        {
            DestroyImmediate(individuals[i].creature);
        }
    }
    
    private void RegenerateCreatures(Individual[] individuals)
    {
        for (int i = 0; i < individuals.Length; i++)
        {
            if(individuals[i].creature != null) Destroy(individuals[i].creature);
            individuals[i].creature = Instantiate(creaturePrefab, spawnPoint, Quaternion.Euler(0f, 0f, 0f));
        }

        ColorizePopulation(individuals);
    }

    private void runNN3(Individual[] individuals)
    {
        foreach (Individual individual in individuals)
        {
            GameObject[] creatureComponents = new []
            {
                individual.creature.transform.GetChild(0).gameObject, // body
                individual.creature.transform.GetChild(1).gameObject, // leg1 (front left)
                individual.creature.transform.GetChild(2).gameObject, // leg2 (rear left)
                individual.creature.transform.GetChild(3).gameObject, // leg3 (front right)
                individual.creature.transform.GetChild(4).gameObject, // leg4 (rear right)
                individual.creature.transform.GetChild(5).gameObject, // leg1* (front left)
                individual.creature.transform.GetChild(6).gameObject, // leg2* (rear left)
                individual.creature.transform.GetChild(7).gameObject, // leg3* (front right)
                individual.creature.transform.GetChild(8).gameObject // leg4* (rear right)
            };
            
            // Get network entries
            float aa = creatureComponents[0].transform.position.y / 4; // body height
            //float bodyRotX = Mathf.InverseLerp(0f, 180f, creatureComponents[0].transform.rotation.eulerAngles.x); // body x rot
            //float bodyRotY = Mathf.InverseLerp(0f, 180f, creatureComponents[0].transform.rotation.eulerAngles.y); // body y rot
            //float bodyRotZ = Mathf.InverseLerp(0f, 180f, creatureComponents[0].transform.rotation.eulerAngles.z); // body Z rot
            float bodyRotX = creatureComponents[0].transform.rotation.eulerAngles.x / 360; // body x rot
            float bodyRotY = creatureComponents[0].transform.rotation.eulerAngles.y / 360; // body x rot
            float bodyRotZ = creatureComponents[0].transform.rotation.eulerAngles.z / 360; // body x rot
            float normalizedCumulatedTime = (cumulatedTime % 4) / 4;
            float a = (creatureComponents[1].transform.rotation.x + 45f) / 90f; // leg1 x rotation
            float b = (creatureComponents[2].transform.rotation.x + 45f) / 90f; // leg2 x rotation
            float c = (creatureComponents[3].transform.rotation.x + 45f) / 90f; // leg3 x rotation
            float d = (creatureComponents[4].transform.rotation.x + 45f) / 90f; // leg4 x rotation
            float a2 = (creatureComponents[5].transform.rotation.x + 45f) / 90f; // leg1 x rotation
            float b2 = (creatureComponents[6].transform.rotation.x + 45f) / 90f; // leg2 x rotation
            float c2 = (creatureComponents[7].transform.rotation.x + 45f) / 90f; // leg3 x rotation
            float d2 = (creatureComponents[8].transform.rotation.x + 45f) / 90f; // leg4 x rotation
            
            var netResults = individual.neuralNetwork.FeedForward(new float[] { aa, bodyRotX, bodyRotY, bodyRotZ, normalizedCumulatedTime });
            //Debug.Log(aa + "," + bodyRotX + "," + bodyRotY + "," + bodyRotZ);
            
            // Force result
            // netResults = new[] {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f};
            // netResults = new[] {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

            // Apply network results
            float[] angles;
            if (symetric)
            {
                angles = new[]
                {
                    (180 * netResults[0] - 90), (180 * netResults[1] - 90), (180 * netResults[0] - 90), (180 * netResults[1] - 90),
                    (180 * netResults[4] - 90), (180 * netResults[5] - 90), (180 * netResults[4] - 90), (180 * netResults[5] - 90)
                };
            }
            else {
                angles = new[]
                {
                    (180 * netResults[0] - 90), (180 * netResults[1] - 90), (180 * netResults[2] - 90), (180 * netResults[3] - 90),
                    (180 * netResults[4] - 90), (180 * netResults[5] - 90), (180 * netResults[6] - 90), (180 * netResults[7] - 90)
                };
            }
            //Debug.Log(angles[0] + " " + angles[1] + " " + angles[2] + " " + angles[3]);
            
            // Travel all legs
            for(int i = 1; i < 9; i++)
            {
                HingeJoint hinge = creatureComponents[i].GetComponent<HingeJoint>();
                var motor = hinge.motor;
                motor.force = 1000000;
                motor.targetVelocity = angles[i-1] * legSpeed;
                motor.freeSpin = false;
                hinge.motor = motor;
                hinge.useMotor = true;
            }
        }
    }

    private NeuralNetwork GenerateRandomNN()
    {
        int minHiddenLayers = hiddenLayers; // Nombre minimum de couches cachées
        int maxHiddenLayers = hiddenLayers; // Nombre maximum de couches cachées
        int minNeuronsPerLayer = 2; // Nombre minimum de neurones par couche cachée
        int maxNeuronsPerLayer = 10; // Nombre maximum de neurones par couche cachée

        // Générer aléatoirement le nombre de couches cachées
        int numHiddenLayers = Random.Range(minHiddenLayers, maxHiddenLayers + 1);

        // Générer aléatoirement le nombre de neurones par couche cachée
        int[] hiddenLayerSizes = new int[numHiddenLayers];
        for (int i = 0; i < numHiddenLayers; i++)
        {
            hiddenLayerSizes[i] = Random.Range(minNeuronsPerLayer, maxNeuronsPerLayer + 1);
        }

        // Définir la taille de la couche de sortie
        int outputSize = 8;
        
        // Définir la fonction d'activation pour les couches cachées
        string hiddenActivation = "sigmoid";

        // Construire le tableau des tailles des couches (entrée, couches cachées, sortie)
        int[] layerSizes = new int[numHiddenLayers + 2];
        layerSizes[0] = 5;
        Array.Copy(hiddenLayerSizes, 0, layerSizes, 1, numHiddenLayers);
        layerSizes[numHiddenLayers + 1] = outputSize;

        // Construire le tableau des fonctions d'activation (couches cachées)
        string[] activationFunctions = new string[numHiddenLayers + 1];
        for (int i = 0; i < numHiddenLayers + 1; i++)
        {
            activationFunctions[i] = hiddenActivation;
        }

        // Créer le réseau de neurones avec les tailles des couches et les fonctions d'activation générées aléatoirement
        NeuralNetwork nn = new NeuralNetwork(layerSizes, activationFunctions);
        return nn;
    }

    public void ColorizePopulation(Individual[] individuals)
    {
        foreach (Individual individual in individuals)
        {
            Material individualMaterial = GenerateRandomMaterial();
            for (int i = 0; i < individual.creature.transform.childCount; i++)
            {
                individual.creature.transform.GetChild(i).GetComponent<Renderer>().material = individualMaterial;
            }
        }
    }
    
    public Material GenerateRandomMaterial()
    {
        Material newMaterial = new Material(Shader.Find("Standard"));
        Color randomColor = new Color(Random.value, Random.value, Random.value);
        newMaterial.color = randomColor;
        return newMaterial;
    }

    private void setGeneratingView()
    {
        status.text = "GENERATING TERRAIN";
        status.color = new Color(1f, 1f, .5f);
        generationText.enabled = false;
        bestFitnessText.enabled = false;
        bestPopulationFitnessText.enabled = false;
        executionFitnessText.text = "";
    }

    private void setTrainingView()
    {
        status.text = "TRAINING";
        status.color = new Color(0.5100101f, 0.5019608f, 1f);
        generationText.enabled = true;
        bestFitnessText.enabled = true;
        bestPopulationFitnessText.enabled = true;
        executionFitnessText.text = "GENERATING TIME : " + generatingTime.ToString("F3") + "s";
    }

    private void setExecutingView()
    {
        status.text = "EXECUTING BEST";
        status.color = new Color(0.5100101f, 0.5019608f, 1f);
        generationText.enabled = true;
        bestFitnessText.enabled = true;
        bestPopulationFitnessText.enabled = true;
        executionFitnessText.text = "";
    }
}
