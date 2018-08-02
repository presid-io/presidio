package main

import (
	"fmt"

	message_types "github.com/Microsoft/presidio-genproto/golang"
	"github.com/Microsoft/presidio/presidio-databinder/cmd/presidio-databinder/cloudstorage"
	"github.com/Microsoft/presidio/presidio-databinder/cmd/presidio-databinder/database"
	databinderInterface "github.com/Microsoft/presidio/presidio-databinder/cmd/presidio-databinder/databinder"
)

func createDatabiner(databinder *message_types.Databinder, kind string) (databinderInterface.DataBinder, error) {
	if isDatabase(kind) {
		if databinder.DbConfig.GetConnectionString() == "" {
			return nil, fmt.Errorf("connectionString var must me set")
		}
		return databaseBinder.New(databinder, kind), nil
	} else if isCloudStorage(kind) {
		return cloudStorageBinder.New(databinder, kind), nil
	}

	return nil, fmt.Errorf("unknown databinder kind")
}

func isDatabase(target string) bool {
	return target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_mssql) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_mysql) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_oracle) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_oracle) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_postgres) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_sqlite3)
}

func isCloudStorage(target string) bool {
	return target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_azureblob) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_s3) ||
		target == message_types.DataBinderTypesEnum.String(message_types.DataBinderTypesEnum_googlestorage)
}
