using System;
using UnityEngine;

public class PlaneRandomHeight : MonoBehaviour
{
    public Mesh meshToDuplicate;
    public MeshCollider meshCol;
    public float PerlinNoiseVal;
    public float Power;

    private Vector3[] verts;
    private Mesh newMesh;

    private void Start()
    {
        newMesh = MeshDuplicator();
        GetComponent<MeshFilter>().mesh = newMesh;
        verts = newMesh.vertices;
        
        for (int i = 0; i < verts.Length; i++)
        {
            float YVal = Mathf.PerlinNoise(verts[i].x * PerlinNoiseVal, verts[i].z * PerlinNoiseVal) * Power;

            verts[i] = new Vector3(verts[i].x, YVal, verts[i].z);
        }

        newMesh.vertices = verts;
        newMesh.RecalculateBounds();
        newMesh.RecalculateNormals();
        meshCol.sharedMesh = newMesh;
    }

    /*
    private void Update()
    {
        for (int i = 0; i < verts.Length; i++)
        {
            float YVal = Mathf.PerlinNoise(verts[i].x * PerlinNoiseVal, verts[i].z * PerlinNoiseVal) * Power;

            verts[i] = new Vector3(verts[i].x, YVal, verts[i].z);
        }

        newMesh.vertices = verts;
        newMesh.RecalculateBounds();
        newMesh.RecalculateNormals();
        meshCol.sharedMesh = newMesh;
    }
    */
    
    
    
    // Function to duplicate a Unity Mesh
    public Mesh MeshDuplicator()
    {
        if (meshToDuplicate == null)
        {
            Debug.LogError("Original mesh is null! Cannot duplicate.");
            return null;
        }

        Mesh duplicateMesh = new Mesh();
        duplicateMesh.vertices = meshToDuplicate.vertices;
        duplicateMesh.triangles = meshToDuplicate.triangles;
        duplicateMesh.normals = meshToDuplicate.normals;
        duplicateMesh.uv = meshToDuplicate.uv;
        duplicateMesh.uv2 = meshToDuplicate.uv2;
        duplicateMesh.uv3 = meshToDuplicate.uv3;
        duplicateMesh.uv4 = meshToDuplicate.uv4;
        duplicateMesh.colors = meshToDuplicate.colors;
        duplicateMesh.colors32 = meshToDuplicate.colors32;
        duplicateMesh.tangents = meshToDuplicate.tangents;

        // Duplicate submeshes if available
        int submeshCount = meshToDuplicate.subMeshCount;
        for (int submeshIndex = 0; submeshIndex < submeshCount; submeshIndex++)
        {
            int[] triangles = meshToDuplicate.GetTriangles(submeshIndex);
            duplicateMesh.SetTriangles(triangles, submeshIndex);
        }

        return duplicateMesh;
    }
}
