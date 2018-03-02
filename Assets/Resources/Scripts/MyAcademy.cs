using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MyAcademy : Academy {

	/*
	 * 
	 * 
	 */
	public override void AcademyReset()
	{


	}

	public override void AcademyStep()
	{
        if (Input.GetMouseButton(0)) {
            if (Time.timeScale == 0f) {
                Time.timeScale = 100f;
            } 

            if (Time.timeScale == 100f) {
                Time.timeScale = 0f;
            }
        }

	}

}
