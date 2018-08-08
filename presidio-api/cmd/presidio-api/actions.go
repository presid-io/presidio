package main

import (
	"fmt"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	log "github.com/Microsoft/presidio/pkg/logger"
	"github.com/Microsoft/presidio/pkg/rpc"

	message_types "github.com/Microsoft/presidio-genproto/golang"
	server "github.com/Microsoft/presidio/pkg/server"
	templates "github.com/Microsoft/presidio/pkg/templates"
)

var analyzeService *message_types.AnalyzeServiceClient
var anonymizeService *message_types.AnonymizeServiceClient
var cronJobService *message_types.CronJobServiceClient

func setupGRPCServices() {

	analyzerSvcHost := os.Getenv("ANALYZER_SVC_HOST")
	if analyzerSvcHost == "" {
		log.Fatal("analyzer service address is empty")
	}

	analyzerSvcPort := os.Getenv("ANALYZER_SVC_PORT")
	if analyzerSvcPort == "" {
		log.Fatal("analyzer service port is empty")
	}

	anonymizerSvcHost := os.Getenv("ANONYMIZER_SVC_HOST")
	if anonymizerSvcHost == "" {
		log.Fatal("anonymizer service address is empty")
	}

	anonymizerSvcPort := os.Getenv("ANONYMIZER_SVC_PORT")
	if anonymizerSvcPort == "" {
		log.Fatal("anonymizer service port is empty")
	}

	// scheduler service port and host will be configured only in case this service is needed.
	schedulerSvcHost := os.Getenv("SCHEDULER_SVC_HOST")
	schedulerSvcPort := os.Getenv("SCHEDULER_SVC_PORT")

	analyzerAddress := analyzerSvcHost + ":" + analyzerSvcPort
	anonymizerAddress := anonymizerSvcHost + ":" + anonymizerSvcPort

	var schedulerAddress string
	if schedulerSvcHost != "" && schedulerSvcPort != "" {
		schedulerAddress = schedulerSvcHost + ":" + schedulerSvcPort
	}
	connectGRPCServices(analyzerAddress, anonymizerAddress, schedulerAddress)

}

func connectGRPCServices(analyzerAddress string, anonymizerAddress string, schedulerAddress string) {
	var err error
	analyzeService, err = rpc.SetupAnalyzerService(analyzerAddress)
	if err != nil {
		log.Error(fmt.Sprintf("Connection to analyzer service failed %q", err))
	}
	anonymizeService, err = rpc.SetupAnonymizeService(anonymizerAddress)
	if err != nil {
		log.Error(fmt.Sprintf("Connection to anonymizer service failed %q", err))
	}

	if schedulerAddress != "" {
		cronJobService, err = rpc.SetupCronJobService(schedulerAddress)
		if err != nil {
			log.Error(fmt.Sprintf("Connection to scheduler service failed %q", err))
		}
	}
}

func (api *API) analyze(c *gin.Context) {
	var analyzeAPIRequest message_types.AnalyzeApiRequest

	if c.Bind(&analyzeAPIRequest) == nil {
		analyzeTemplate := analyzeAPIRequest.AnalyzeTemplate

		if analyzeTemplate == nil && analyzeAPIRequest.AnalyzeTemplateId != "" {
			analyzeTemplate = &message_types.AnalyzeTemplate{}
			api.getTemplate(c.Param("project"), "analyze", analyzeAPIRequest.AnalyzeTemplateId, analyzeTemplate, c)
		} else if analyzeTemplate == nil {
			c.AbortWithError(http.StatusBadRequest, fmt.Errorf("AnalyzeTemplate or AnalyzeTemplateId must be supplied"))
			return
		}

		res := api.invokeAnalyze(analyzeTemplate, analyzeAPIRequest.Text, c)
		if res == nil {
			return
		}
		server.WriteResponse(c, http.StatusOK, res)
	}
}

func isTemplateIDInitialized(template *message_types.AnalyzeTemplate, templateID string) bool {
	return template == nil && templateID != ""
}

func (api *API) anonymize(c *gin.Context) {
	var anonymizeAPIRequest message_types.AnonymizeApiRequest

	if c.Bind(&anonymizeAPIRequest) == nil {
		var err error
		analyzeTemplate := anonymizeAPIRequest.AnalyzeTemplate
		id := anonymizeAPIRequest.AnalyzeTemplateId
		project := c.Param("project")

		if isTemplateIDInitialized(analyzeTemplate, anonymizeAPIRequest.AnalyzeTemplateId) {
			analyzeTemplate = &message_types.AnalyzeTemplate{}
			api.getTemplate(project, "analyze", id, analyzeTemplate, c)
		} else if analyzeTemplate == nil {
			c.AbortWithError(http.StatusBadRequest, fmt.Errorf("AnalyzeTemplate or AnalyzeTemplateId must be supplied"))
			return
		}

		analyzeRes := api.invokeAnalyze(analyzeTemplate, anonymizeAPIRequest.Text, c)
		if analyzeRes == nil {
			return
		}

		anonymizeTemplate := anonymizeAPIRequest.AnonymizeTemplate
		if anonymizeTemplate == nil && anonymizeAPIRequest.AnonymizeTemplateId != "" {
			id = anonymizeAPIRequest.AnonymizeTemplateId
			anonymizeTemplate = &message_types.AnonymizeTemplate{}
			api.getTemplate(project, "anonymize", id, anonymizeTemplate, c)
			if err != nil {
				c.AbortWithError(http.StatusBadRequest, err)
			}
		} else if anonymizeTemplate == nil {
			c.AbortWithError(http.StatusBadRequest, fmt.Errorf("AnalyzeTemplate or AnalyzeTemplateId must be supplied"))
			return
		}

		anonymizeRes := api.invokeAnonymize(anonymizeTemplate, anonymizeAPIRequest.Text, analyzeRes.AnalyzeResults, c)
		if anonymizeRes == nil {
			return
		}
		server.WriteResponse(c, http.StatusOK, anonymizeRes)
	}
}

func (api *API) scheduleCronJob(c *gin.Context) {
	var cronAPIJobRequest message_types.CronJobApiRequest

	if c.Bind(&cronAPIJobRequest) == nil {
		project := c.Param("project")
		scheulderResponse := api.invokeCronJobScheduler(cronAPIJobRequest, project, c)
		if scheulderResponse == nil {
			return
		}

		server.WriteResponse(c, http.StatusOK, scheulderResponse)
	}
}

func (api *API) invokeCronJobScheduler(cronJobAPIRequest message_types.CronJobApiRequest, project string, c *gin.Context) *message_types.CronJobResponse {
	cronJobTemplate := &message_types.CronJobTemplate{}
	api.getTemplate(project, "schedule-cronjob", cronJobAPIRequest.CronJobTemplateId, cronJobTemplate, c)

	scanID := cronJobTemplate.ScanTemplateId
	scanTemplate := &message_types.ScanTemplate{}
	api.getTemplate(project, "scan", scanID, scanTemplate, c)

	dataSyncTemplate := &message_types.DataSyncTemplate{}
	api.getTemplate(project, "dataSync", cronJobTemplate.DataSyncTemplateId, dataSyncTemplate, c)

	analyzeTemplate := &message_types.AnalyzeTemplate{}
	api.getTemplate(project, "analyze", cronJobTemplate.AnalyzeTemplateId, analyzeTemplate, c)

	anonymizeTemplate := &message_types.AnonymizeTemplate{}
	if cronJobTemplate.AnonymizeTemplateId != "" {
		api.getTemplate(project, "anonymize", cronJobTemplate.AnonymizeTemplateId, anonymizeTemplate, c)
	}

	scanRequest := &message_types.ScanRequest{
		AnalyzeTemplate:    analyzeTemplate,
		AnonymizeTemplate:  anonymizeTemplate,
		DataSyncTemplate:   dataSyncTemplate,
		CloudStorageConfig: scanTemplate.CloudStorageConfig,
		Kind:               scanTemplate.Kind,
		MinProbability:     scanTemplate.MinProbability,
	}

	request := &message_types.CronJobRequest{
		Name:        cronJobTemplate.Name,
		Description: cronJobTemplate.Description,
		Trigger:     cronJobTemplate.Trigger,
		ScanRequest: scanRequest,
	}
	srv := *cronJobService
	res, err := srv.Apply(c, request)
	if err != nil {
		c.AbortWithError(http.StatusInternalServerError, err)
		return nil
	}
	return res
}

func (api *API) invokeAnonymize(anonymizeTemplate *message_types.AnonymizeTemplate, text string, results []*message_types.AnalyzeResult, c *gin.Context) *message_types.AnonymizeResponse {
	srv := *anonymizeService

	request := &message_types.AnonymizeRequest{
		Template:       anonymizeTemplate,
		Text:           text,
		AnalyzeResults: results,
	}
	res, err := srv.Apply(c, request)
	if err != nil {
		c.AbortWithError(http.StatusInternalServerError, err)
		return nil
	}
	return res
}

func (api *API) invokeAnalyze(analyzeTemplate *message_types.AnalyzeTemplate, text string, c *gin.Context) *message_types.AnalyzeResponse {
	analyzeRequest := &message_types.AnalyzeRequest{
		AnalyzeTemplate: analyzeTemplate,
		Text:            text,
	}

	srv := *analyzeService
	analyzeResponse, err := srv.Apply(c, analyzeRequest)
	if err != nil {
		c.AbortWithError(http.StatusInternalServerError, err)
		return nil
	}

	return analyzeResponse
}

func (api *API) getTemplate(project string, action string, id string, obj interface{}, c *gin.Context) {
	key := templates.CreateKey(project, action, id)
	template, err := api.templates.GetTemplate(key)
	if err != nil {
		c.AbortWithError(http.StatusBadRequest, err)
	}
	err = templates.ConvertJSONToInterface(template, obj)
	if err != nil {
		c.AbortWithError(http.StatusBadRequest, err)
	}
}
