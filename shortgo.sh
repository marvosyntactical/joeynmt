#!/bin/bash

name=$1

bash slurmgo.sh $name 0-08:00:00 128000 gpushort
