package kube

import (
	"testing"

	"github.com/stretchr/testify/assert"
	apiv1 "k8s.io/api/core/v1"

	"github.com/Microsoft/presidio/pkg/platform"

	"k8s.io/client-go/kubernetes/fake"
)

func TestCreateAndDeleteJob(t *testing.T) {

	client := fake.NewSimpleClientset()

	store := &store{
		client:    client,
		namespace: "default",
	}

	name := "jobName"

	containerDetails := []platform.ContainerDetails{
		{
			EnvVars: []apiv1.EnvVar{
				{Name: "envVar1", Value: "envVarval"},
			},
			Name:            "jobName",
			Image:           "imageName",
			ImagePullPolicy: apiv1.PullAlways,
		},
	}

	// Create job
	err := store.CreateJob(name, containerDetails)
	if err != nil {
		t.Fatal(err)
	}

	// List jobs
	jobs, _ := store.ListJobs()
	assert.Equal(t, len(jobs), 1)
	assert.Equal(t, jobs[0], name)

	// Delete job
	err = store.DeleteJob(name)
	if err != nil {
		t.Fatal(err)
	}

	// List jobs
	jobs, _ = store.ListJobs()
	assert.Equal(t, len(jobs), 0)
}
