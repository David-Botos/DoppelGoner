// components/EntityDetails/EntityDetailsCard.tsx
import React from 'react';
import { Entity, EntityDetails } from '../../lib/types';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  Divider, 
  Link, 
  CircularProgress, 
  Box
} from '@mui/material';
import PlaceIcon from '@mui/icons-material/Place';
import PhoneIcon from '@mui/icons-material/Phone';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import LanguageIcon from '@mui/icons-material/Language';
import MiscellaneousServicesIcon from '@mui/icons-material/MiscellaneousServices';

interface EntityDetailsCardProps {
  entity: Entity;
  entityDetails: EntityDetails | undefined;
  loading?: boolean;
  title?: string;
  elevation?: number;
}

const EntityDetailsCard: React.FC<EntityDetailsCardProps> = ({ 
  entity, 
  entityDetails, 
  loading = false,
  title,
  elevation = 1
}) => {
  if (!entity) return null;
  
  return (
    <Card elevation={elevation}>
      <CardHeader
        title={title || entity.name}
        titleTypographyProps={{ variant: 'h6' }}
        sx={{ pb: 1 }}
      />
      
      <Divider />
      
      <CardContent sx={{ pt: 2 }}>
        {loading ? (
          <Box display="flex" alignItems="center" px={1} py={2}>
            <CircularProgress size={20} sx={{ mr: 2 }} />
            <Typography variant="body2">Loading details...</Typography>
          </Box>
        ) : entityDetails ? (
          <>
            {entityDetails.organization_url && (
              <Box display="flex" alignItems="center" mb={2}>
                <LanguageIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="body2" component="span" sx={{ mr: 1, fontWeight: 500 }}>
                  URL:
                </Typography>
                <Link href={entityDetails.organization_url} target="_blank" underline="hover">
                  <Typography variant="body2" noWrap>
                    {entityDetails.organization_url}
                  </Typography>
                </Link>
              </Box>
            )}
            
            {entityDetails.phones && entityDetails.phones.length > 0 && (
              <Box mb={2}>
                <Box display="flex" alignItems="center" mb={0.5}>
                  <PhoneIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    Phones:
                  </Typography>
                </Box>
                <List dense disablePadding>
                  {entityDetails.phones.map((phone, index) => (
                    <ListItem key={index} sx={{ py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2">
                            {phone.number}
                            {phone.extension && ` ext. ${phone.extension}`}
                            {phone.type && ` (${phone.type})`}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
            
            {entityDetails.addresses && entityDetails.addresses.length > 0 && (
              <Box mb={2}>
                <Box display="flex" alignItems="center" mb={0.5}>
                  <PlaceIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    Addresses:
                  </Typography>
                </Box>
                <List dense disablePadding>
                  {entityDetails.addresses.map((address, index) => (
                    <ListItem key={index} sx={{ py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2">
                            {address.address_1}
                            {address.address_2 && `, ${address.address_2}`}
                            {`, ${address.city}, ${address.state_province} ${address.postal_code}`}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
            
            {entityDetails.locations && entityDetails.locations.length > 0 && (
              <Box mb={2}>
                <Box display="flex" alignItems="center" mb={0.5}>
                  <LocationOnIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    Coordinates:
                  </Typography>
                </Box>
                <List dense disablePadding>
                  {entityDetails.locations.map((location, index) => (
                    <ListItem key={index} sx={{ py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center">
                            <Typography variant="body2" sx={{ mr: 1 }}>
                              {location.latitude}, {location.longitude}
                            </Typography>
                            <Link 
                              href={`https://www.latlong.net/c/?lat=${location.latitude}&long=${location.longitude}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              underline="hover"
                              color="primary"
                              variant="body2"
                            >
                              View on map
                            </Link>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
            
            {entityDetails.services && entityDetails.services.length > 0 && (
              <Box mb={1}>
                <Box display="flex" alignItems="center" mb={0.5}>
                  <MiscellaneousServicesIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    Services:
                  </Typography>
                </Box>
                <List dense disablePadding>
                  {entityDetails.services.map((service) => (
                    <ListItem key={service.id} sx={{ py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ fontWeight: service.short_description ? 500 : 400 }}>
                            {service.name}
                          </Typography>
                        }
                        secondary={service.short_description && (
                          <Typography variant="body2" color="text.secondary">
                            {service.short_description}
                          </Typography>
                        )}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </>
        ) : (
          <Typography variant="body2" color="text.secondary" px={1}>
            No details available.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default EntityDetailsCard;