using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MyAgent : Agent {

    public float rotationSpeed = 10f;
    private GameObject ball;
    private Rigidbody rb;

    public override void InitializeAgent() {
        ball = GameObject.FindWithTag("Ball");
        rb = GetComponent<Rigidbody>();
    }

    public override List<float> CollectState() {

        List<float> state = new List<float>();
        //state.Add(transform.rotation.x);
        state.Add(transform.rotation.z);
        state.Add(ball.transform.position.x);
        state.Add(ball.transform.position.y);
        //state.Add(ball.transform.position.z);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.x);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.y);
        //state.Add(ball.transform.GetComponent<Rigidbody>().velocity.z);
        Debug.Log(state[3]);
        Debug.Log(state[4]);

        return state;
	}

    public override void AgentStep(float[] act) {

        if (Input.anyKey) {

            float hor = Input.GetAxis("Horizontal");
            float vert = Input.GetAxis("Vertical");
            transform.Rotate(new Vector3(1, 0, 0), hor * rotationSpeed);
            transform.Rotate(new Vector3(0, 0, 1), vert * rotationSpeed);

            reward = -999;

            if (ball.transform.position.y < 0f) {
                done = true;
            }

        } else {

            if (act[0] == 0) {
                
            //} else if (act[0] == 1) {
            //    transform.Rotate(new Vector3(1, 0, 0), rotationSpeed);
            //} else if (act[0] == 2) {
            //    transform.Rotate(new Vector3(-1, 0, 0), rotationSpeed);
            } else if (act[0] == 1) {
                transform.Rotate(new Vector3(0, 0, 1), rotationSpeed);
            } else if (act[0] == 2) {
                transform.Rotate(new Vector3(0, 0, -1), rotationSpeed);
            } 

            if (done == false) {
                reward = 10f;
            }

            if (ball.transform.position.y < 0f) {
                done = true;
                reward = -100f;
            }
        }
    }

	public override void AgentReset() {
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.position = new Vector3(0, 0, 0);
        transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
//        transform.Rotate(new Vector3(1, 0, 0), Random.Range(-3f, 3f));
 //       transform.Rotate(new Vector3(0, 0, 1), Random.Range(-3f, 3f));
        ball.GetComponent<Rigidbody>().velocity = Vector3.zero;
        ball.transform.position = new Vector3(0.3f, 1f, 0f) + gameObject.transform.position;
        Debug.Log(rb.velocity);

    }

    public override void AgentOnDone() {

	}
}
