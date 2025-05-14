// components/TaskDetails/index.tsx
import React from "react";
import { EntityGroup, EntityDetails, Entity } from "../../lib/types";
import ActionButtons from "./ActionButtons";
import {
  Box,
  Typography,
  Paper,
  Alert,
  Stack,
  useTheme,
} from "@mui/material";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import EntityDetailsCard from "../EntityDetails/EntityDetailsCard";
import MatchEvidencePanel from "../TaskDetails/MatchEvidencePanel";

interface TaskDetailsProps {
  task: EntityGroup | null;
  entityDetailsCache: Record<string, EntityDetails>;
  onConfirm: () => void;
  onReject: () => void;
  submitting: boolean;
  error: Error | null;
}

const TaskDetails: React.FC<TaskDetailsProps> = ({
  task,
  entityDetailsCache,
  onConfirm,
  onReject,
  submitting,
  error,
}) => {
  const theme = useTheme();

  if (!task) {
    return (
      <Box p={4} textAlign="center" color="text.secondary">
        <Typography variant="body1">
          No task selected. Select a cluster to begin reviewing.
        </Typography>
      </Box>
    );
  }

  const sourceDetails = entityDetailsCache[task.source];
  const targetDetails = entityDetailsCache[task.target];

  // Create proper entity objects with all required fields
  const sourceEntity: Entity = {
    id: task.source,
    name: sourceDetails?.organization_name || "Loading...",
    // Using the task.source as the organization_id since it's the identifier we have
    organization_id: task.source,
  };

  const targetEntity: Entity = {
    id: task.target,
    name: targetDetails?.organization_name || "Loading...",
    organization_id: task.target,
  };

  return (
    <Box
      p={3}
      height="100%"
      overflow="auto"
      bgcolor={theme.palette.background.default}
    >
      <Box mb={3}>
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          Entity Link Review
        </Typography>
      </Box>

      {/* <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Box display="flex" alignItems="center">
            <Typography variant="subtitle1" fontWeight="medium" mr={1}>
              Match Method:
            </Typography>
            <Chip 
              label={task.method_type} 
              color="primary" 
              size="small"
              sx={{ 
                backgroundColor: theme.palette.primary.light,
                color: theme.palette.primary.dark
              }} 
            />
          </Box>
          <Box>
            <Typography variant="subtitle1" fontWeight="medium" component="span" mr={1}>
              Confidence:
            </Typography>
            <Chip 
              label={task.confidence_score.toFixed(2)} 
              size="small"
              color={task.confidence_score > 0.75 ? "success" : "warning"}
            />
          </Box>
        </Box>
      </Paper> */}

      {/* Replace Grid with Stack/Box layout */}
      <Box mb={4}>
        <Typography
          variant="subtitle1"
          fontWeight="bold"
          mb={1}
          display="flex"
          alignItems="center"
        >
          <CompareArrowsIcon sx={{ mr: 1 }} />
          Match Evidence
        </Typography>
        <Paper elevation={2} sx={{ p: 2 }}>
          <MatchEvidencePanel task={task} />
        </Paper>
      </Box>

     

      <Box mb={3}>
        <Typography variant="subtitle1" fontWeight="bold" mb={1}>
          Decision
        </Typography>
        <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
        <Typography variant="body2" color="text.secondary" mb={2}>
          Do these two entities represent the same real-world organization?
        </Typography>

        <ActionButtons
          onConfirm={onConfirm}
          onReject={onReject}
          submitting={submitting}
        />
        </Paper>
      </Box>

      <Stack direction={{ xs: "column", md: "row" }} spacing={3} sx={{ mb: 3 }}>
        
        <Box flex={1}>
          <Typography variant="subtitle1" fontWeight="bold" mb={1}>
            Entity 1
          </Typography>
          <EntityDetailsCard
            entity={sourceEntity}
            entityDetails={sourceDetails}
            loading={!sourceDetails}
            title={`${sourceDetails?.organization_name || "Loading..."}`}
          />
        </Box>

        <Box flex={1}>
          <Typography variant="subtitle1" fontWeight="bold" mb={1}>
            Entity 2
          </Typography>
          <EntityDetailsCard
            entity={targetEntity}
            entityDetails={targetDetails}
            loading={!targetDetails}
            title={`${targetDetails?.organization_name || "Loading..."}`}
          />
        </Box>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error.message}
        </Alert>
      )}
    </Box>
  );
};

export default TaskDetails;
