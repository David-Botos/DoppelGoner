// components/TaskDetails/MatchEvidencePanel.tsx
import React from 'react';
import { EntityGroup } from '../../lib/types';
import { 
  Box, 
  Typography, 
  Divider,
  Stack,
  useTheme
} from '@mui/material';

interface MatchEvidencePanelProps {
  task: EntityGroup;
}

const MatchEvidencePanel: React.FC<MatchEvidencePanelProps> = ({ task }) => {
  const theme = useTheme();
  
  // Helper function to safely render values (handle empty objects and nulls)
  const safeRender = (value: unknown): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'object' && Object.keys(value).length === 0) return 'N/A';
    return String(value);
  };
  
  if (!task.match_values || !task.match_values.values) {
    return <Typography color="text.secondary">No match evidence available</Typography>;
  }
  
  // Determine which type of match method we have
  const renderContent = () => {
    const { match_values } = task;
    
    switch (task.method_type) {
      case 'email':
        return (
          <>
            <Stack 
              direction={{ xs: 'column', md: 'row' }}
              spacing={3}
            >
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 1 Email
                </Typography>
                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                  {safeRender(match_values.values.original_email1)}
                </Typography>
              </Box>
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 2 Email
                </Typography>
                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                  {safeRender(match_values.values.original_email2)}
                </Typography>
              </Box>
            </Stack>
            
            <Divider sx={{ my: 2 }} />
            
            <Box bgcolor={theme.palette.success.light} p={2} borderRadius={1}>
              <Typography variant="subtitle2" fontWeight="medium">
                Normalized Shared Email
              </Typography>
              <Typography variant="body1">
                {safeRender(match_values.values.normalized_shared_email)}
              </Typography>
            </Box>
          </>
        );
        
      case 'name':
        return (
          <>
            <Stack 
              direction={{ xs: 'column', md: 'row' }}
              spacing={3}
            >
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 1 Name
                </Typography>
                <Typography variant="body1">
                  {safeRender(match_values.values.original_name1)}
                </Typography>
              </Box>
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 2 Name
                </Typography>
                <Typography variant="body1">
                  {safeRender(match_values.values.original_name2)}
                </Typography>
              </Box>
            </Stack>
            
            <Divider sx={{ my: 2 }} />
            
            <Box bgcolor={theme.palette.success.light} p={2} borderRadius={1}>
              <Typography variant="subtitle2" fontWeight="medium">
                Name Similarity Score
              </Typography>
              <Typography variant="body1">
                {((match_values.values.pre_rl_similarity_score as number) || 0).toFixed(2)}
              </Typography>
            </Box>
          </>
        );
        
      case 'url':
        return (
          <>
            <Stack 
              direction={{ xs: 'column', md: 'row' }}
              spacing={3}
            >
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 1 URL
                </Typography>
                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                  {safeRender(match_values.values.original_url1)}
                </Typography>
              </Box>
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 2 URL
                </Typography>
                <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                  {safeRender(match_values.values.original_url2)}
                </Typography>
              </Box>
            </Stack>
            
            <Divider sx={{ my: 2 }} />
            
            <Box bgcolor={theme.palette.success.light} p={2} borderRadius={1}>
              <Typography variant="subtitle2" fontWeight="medium">
                Normalized Shared URL
              </Typography>
              <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                {safeRender(match_values.values.normalized_shared_url)}
              </Typography>
            </Box>
          </>
        );
        
      case 'phone':
        return (
          <>
            <Stack 
              direction={{ xs: 'column', md: 'row' }}
              spacing={3}
            >
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 1 Phone
                </Typography>
                <Typography variant="body1">
                  {safeRender(match_values.values.original_phone1)}
                </Typography>
              </Box>
              <Box flex={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Organization 2 Phone
                </Typography>
                <Typography variant="body1">
                  {safeRender(match_values.values.original_phone2)}
                </Typography>
              </Box>
            </Stack>
            
            <Divider sx={{ my: 2 }} />
            
            <Box bgcolor={theme.palette.success.light} p={2} borderRadius={1}>
              <Typography variant="subtitle2" fontWeight="medium">
                Normalized Shared Phone
              </Typography>
              <Typography variant="body1">
                {safeRender(match_values.values.normalized_shared_phone)}
              </Typography>
            </Box>
          </>
        );
        
      default:
        return (
          <Box>
            <Typography variant="subtitle2" mb={1}>
              Match Evidence (JSON):
            </Typography>
            <Box 
              component="pre" 
              p={2} 
              bgcolor={theme.palette.grey[100]} 
              borderRadius={1} 
              overflow="auto"
              fontSize="0.8rem"
            >
              {JSON.stringify(match_values.values, null, 2)}
            </Box>
          </Box>
        );
    }
  };

  return renderContent();
};

export default MatchEvidencePanel;